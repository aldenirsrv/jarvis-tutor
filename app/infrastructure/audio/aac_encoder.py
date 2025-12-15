from __future__ import annotations

import os
import select
import shutil
import subprocess
from typing import Generator, Iterable
from app.domain.entities.audio import AudioFrame
import logging

logger = logging.getLogger(__name__)


def _drain_non_blocking(pipe_fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([pipe_fd], [], [], 0)
        if not ready:
            break
        try:
            data = os.read(pipe_fd, 4096)
        except BlockingIOError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def aac_stream(frames: Iterable[AudioFrame], *, container: str = "adts") -> Generator[bytes, None, None]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (needed for AAC streaming)")

    frame_iter = iter(frames)
    try:
        first = next(frame_iter)
    except StopIteration:
        return

    ar = str(first.sample_rate)
    ac = str(first.channels)
    if container == "mp4":
        cmd = [
            "ffmpeg","-loglevel","error",
            "-f","s16le","-ar",ar,"-ac",ac,"-i","pipe:0",
            "-c:a","aac","-b:a","64k",
            "-movflags","+frag_keyframe+empty_moov+default_base_moof",
            "-f","mp4","pipe:1",
        ]
    else:
        cmd = [
            "ffmpeg","-loglevel","error",
            "-f","s16le","-ar",ar,"-ac",ac,"-i","pipe:0",
            "-c:a","aac","-b:a","64k","-f","adts","pipe:1",
        ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Non-blocking read using select + os.read
    os.set_blocking(proc.stdout.fileno(), False)
    out_fd = proc.stdout.fileno()

    try:
        try:
            proc.stdin.write(first.data)
            proc.stdin.flush()
            logger.info("aac_encoder: wrote first frame | bytes=%d", len(first.data))
            out = _drain_non_blocking(out_fd)
            if out:
                yield out

            for frame in frame_iter:
                try:
                    proc.stdin.write(frame.data)
                    proc.stdin.flush()
                    logger.info("aac_encoder: wrote frame | bytes=%d", len(frame.data))
                    out = _drain_non_blocking(out_fd)
                    if out:
                        yield out
                except BrokenPipeError as exc:
                    logger.error("aac_encoder: broken pipe while writing to ffmpeg: %s", exc)
                    raise RuntimeError("AAC encode failed: broken pipe") from exc
        except Exception as exc:
            # Ensure ffmpeg is torn down if upstream fails
            try:
                proc.stdin.close()
            finally:
                proc.terminate()
            raise RuntimeError(f"AAC encode failed: {exc}") from exc

        # Finish cleanly
        try:
            proc.stdin.close()
        except Exception:
            pass
        os.set_blocking(out_fd, True)
        total_out = 0
        while True:
            out = proc.stdout.read(4096)
            if not out:
                break
            total_out += len(out)
            yield out
        logger.info("aac_encoder: finished draining ffmpeg | total_bytes=%d", total_out)
        try:
            ret = proc.wait(timeout=5)
            logger.info("aac_encoder: ffmpeg exited | code=%s", ret)
        except subprocess.TimeoutExpired:
            logger.warning("aac_encoder: ffmpeg did not exit in time; terminating")
            proc.terminate()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
