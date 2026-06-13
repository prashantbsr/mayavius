import type { JobHandle, JobStatus, Mv4dScene } from "@/types";
import { decodeReconstruction } from "@/lib/wire/decoder";
import {
  API_BASE_URL,
  POLL_INTERVAL_MS,
  SSE_WATCHDOG_MS,
  USE_SSE,
} from "@/config";

// Thin client for the FastAPI reconstruction backend (async job model,
// handover §4.4): submit a clip → poll/stream status → fetch the binary result.
// Endpoint paths, payloads and the error contract are defined in
// spec/06-backend-spec.md. Base URL: API_BASE_URL in config.ts.
//
// W2 (this task) implements submit/poll/SSE on top of the W0 `fetchResult`
// (the wire-format seam — it decodes the MV4D blob into an `Mv4dScene`,
// spec/05 §5.2). The SSE consumption contract is spec/07 §6 step 2 and is
// load-bearing: the backend tags every event `event=<status>` (a NAMED SSE
// event), so we MUST `addEventListener("queued"|"running"|"done"|"failed", …)`
// — `EventSource.onmessage` (unnamed `message`) will NEVER fire.

/** The poll/SSE JSON the backend emits (`job_to_json`, spec/06 §6) — snake_case.
 * The six base keys are always present; `result`/`error` appear iff done/failed. */
interface PollJson {
  id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  adapter_id: string;
  weights_license: string;
  /** relative URL string, present iff status==="done". */
  result?: string;
  /** present iff status==="failed". */
  error?: { code: string; message: string };
}

/** The terminal/progress statuses the backend tags as NAMED SSE events
 * (`event=<status>`, spec/06 §7). We listen for exactly these. */
const SSE_EVENT_NAMES: JobStatus[] = ["queued", "running", "done", "failed"];

/** Map a `job_to_json` payload (snake_case) → the frontend `JobHandle`. */
function pollJsonToHandle(j: PollJson): JobHandle {
  return { id: j.id, status: j.status, progress: j.progress };
}

/** Callbacks the viewer's on-mount loader passes to `streamJob` (spec/07 §6). */
export interface StreamHandlers {
  /** Fired on every non-terminal update with the parsed job state. */
  onProgress?: (handle: JobHandle, json: PollJson) => void;
  /** Fired once on `done` with the decoded scene (after `fetchResult`). */
  onDone?: (scene: Mv4dScene, json: PollJson) => void;
  /** Fired once on `failed`, on a decode/fetch failure, or a transport error. */
  onError?: (message: string, json?: PollJson) => void;
}

/**
 * Submit a clip for reconstruction. POST `multipart/form-data` (field `clip`)
 * to `{API_BASE_URL}/jobs`; the backend returns **202** `{ job_id, status, … }`
 * (spec/06 §7). We surface a `JobHandle` whose `id` is that `job_id` and whose
 * status is `"queued"` (the value the backend sets on accept). The `view/[id]`
 * route then loads it (spec/07 §6 step 1).
 */
export async function submitClip(file: File): Promise<JobHandle> {
  const form = new FormData();
  // Field name MUST be "clip" (spec/06 §7 POST /jobs).
  form.append("clip", file);
  const res = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(`submitClip: POST /jobs failed (${res.status})`);
  }
  const body = (await res.json()) as { job_id: string; status?: JobStatus };
  return { id: body.job_id, status: "queued" };
}

/**
 * Poll a job's status once. `GET {API_BASE_URL}/jobs/{id}` returns the
 * `job_to_json` shape (spec/06 §7); we map it to a `JobHandle`.
 */
export async function getJobStatus(jobId: string): Promise<JobHandle> {
  const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`getJobStatus: GET /jobs/${jobId} failed (${res.status})`);
  }
  const json = (await res.json()) as PollJson;
  return pollJsonToHandle(json);
}

/**
 * Subscribe to a job's progress and resolve its result.
 *
 * Primary path = SSE: open an `EventSource` on `{API_BASE_URL}/jobs/{id}/stream`
 * and `addEventListener` for the NAMED events `queued|running|done|failed`
 * (spec/07 §6 step 2 — `onmessage` never fires). Each handler `JSON.parse`s
 * `e.data` into the poll-JSON shape, forwards progress, and on `done` calls
 * `fetchResult` (decode → `onDone`); on `failed` surfaces the error.
 *
 * Fallback = polling `GET /jobs/{id}` every `POLL_INTERVAL_MS` — used when
 * `USE_SSE===false`, on `EventSource.onerror`, OR if no event arrives within
 * `SSE_WATCHDOG_MS` (the watchdog guards a silently-stalled stream).
 *
 * Returns a cleanup function (close the stream / stop the poll loop).
 */
export function streamJob(jobId: string, handlers: StreamHandlers): () => void {
  let stopped = false;
  let source: EventSource | null = null;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let watchdog: ReturnType<typeof setTimeout> | null = null;

  const clearWatchdog = () => {
    if (watchdog != null) {
      clearTimeout(watchdog);
      watchdog = null;
    }
  };

  const cleanup = () => {
    stopped = true;
    clearWatchdog();
    if (pollTimer != null) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    if (source != null) {
      source.close();
      source = null;
    }
  };

  // Terminal handling shared by SSE + polling: decode on done, surface on fail.
  const handleTerminal = async (json: PollJson): Promise<void> => {
    if (stopped) return;
    if (json.status === "done") {
      try {
        const scene = await fetchResult(jobId);
        if (stopped) return;
        handlers.onDone?.(scene, json);
      } catch (err) {
        if (stopped) return;
        handlers.onError?.(
          err instanceof Error ? err.message : String(err),
          json,
        );
      }
    } else if (json.status === "failed") {
      const msg = json.error?.message ?? "reconstruction failed";
      handlers.onError?.(msg, json);
    }
    cleanup();
  };

  // ── Polling fallback ────────────────────────────────────────────────────────
  const startPolling = () => {
    if (stopped) return;
    // We may arrive here from an SSE error/stall — make sure the stream is gone.
    clearWatchdog();
    if (source != null) {
      source.close();
      source = null;
    }
    const tick = async () => {
      if (stopped) return;
      try {
        const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
        if (stopped) return;
        if (!res.ok) {
          handlers.onError?.(
            `poll: GET /jobs/${jobId} failed (${res.status})`,
          );
          cleanup();
          return;
        }
        const json = (await res.json()) as PollJson;
        if (stopped) return;
        if (json.status === "done" || json.status === "failed") {
          await handleTerminal(json);
          return;
        }
        handlers.onProgress?.(pollJsonToHandle(json), json);
        pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) {
        if (stopped) return;
        handlers.onError?.(err instanceof Error ? err.message : String(err));
        cleanup();
      }
    };
    void tick();
  };

  // ── SSE primary ─────────────────────────────────────────────────────────────
  const startSse = () => {
    let fellBack = false;
    const fallBack = () => {
      if (stopped || fellBack) return;
      fellBack = true;
      startPolling();
    };

    // Watchdog: a stream that connects but never emits an event is a silent
    // stall — switch to polling after SSE_WATCHDOG_MS of silence (spec/07 §6).
    const armWatchdog = () => {
      clearWatchdog();
      watchdog = setTimeout(fallBack, SSE_WATCHDOG_MS);
    };

    try {
      source = new EventSource(`${API_BASE_URL}/jobs/${jobId}/stream`);
    } catch {
      // EventSource unavailable / threw synchronously → poll instead.
      startPolling();
      return;
    }

    const onNamedEvent = (e: MessageEvent) => {
      if (stopped) return;
      // An event arrived — the stream is live; reset the silence watchdog.
      armWatchdog();
      let json: PollJson;
      try {
        json = JSON.parse(e.data) as PollJson;
      } catch {
        // Malformed frame: don't tear down on a single bad event, just ignore.
        return;
      }
      if (json.status === "done" || json.status === "failed") {
        clearWatchdog();
        void handleTerminal(json);
        return;
      }
      handlers.onProgress?.(pollJsonToHandle(json), json);
    };

    // Named events only — onmessage (unnamed "message") never fires (§6 step 2).
    for (const name of SSE_EVENT_NAMES) {
      source.addEventListener(name, onNamedEvent as EventListener);
    }

    // Transport error (connection refused, server closed, etc.) → poll.
    source.onerror = () => {
      if (stopped) return;
      // EventSource auto-reconnects on transient drops; once the work is done
      // the server closes the stream and we'd see an error with no more data.
      // Falling back to a single poll resolves the terminal state cleanly.
      fallBack();
    };

    armWatchdog();
  };

  if (USE_SSE && typeof EventSource !== "undefined") {
    startSse();
  } else {
    startPolling();
  }

  return cleanup;
}

/**
 * Fetch the binary MV4D v1 result for a completed job and decode it into an
 * `Mv4dScene` (spec/05 §5.2). `GET {API_BASE_URL}/jobs/{id}/result` returns the
 * raw `ArrayBuffer` (spec/06 §7); `decodeReconstruction` throws `Mv4dDecodeError`
 * on a malformed buffer (spec/05 §8).
 */
export async function fetchResult(jobId: string): Promise<Mv4dScene> {
  const res = await fetch(`${API_BASE_URL}/jobs/${jobId}/result`);
  if (!res.ok) {
    throw new Error(
      `fetchResult: GET /jobs/${jobId}/result failed (${res.status})`,
    );
  }
  const buffer = await res.arrayBuffer();
  return decodeReconstruction(buffer);
}
