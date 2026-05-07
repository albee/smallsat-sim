from dataclasses import dataclass


@dataclass(frozen=True)
class EpochTiming:
    rollout: float
    scan: float
    buffer: float
    update: float
    logging: float
    eval: float
    save_ckpt: float
    total: float

    @property
    def other_rollout(self) -> float:
        return max(0.0, self.rollout - (self.scan + self.buffer))


def format_epoch_timing_line(
    *,
    global_epoch: int,
    total_epochs: int,
    phase_name: str,
    phase_epoch: int,
    phase_epochs: int,
    timing: EpochTiming,
) -> str:
    return (
        f"[Timing] Epoch {global_epoch}/{total_epochs} "
        f"(phase={phase_name} {phase_epoch}/{phase_epochs}): "
        f"rollout {timing.rollout:.2f}s "
        f"(scan {timing.scan:.2f}s, buffer {timing.buffer:.2f}s, other {timing.other_rollout:.2f}s) | "
        f"update {timing.update:.2f}s | "
        f"logging {timing.logging:.2f}s | "
        f"eval {timing.eval:.2f}s | "
        f"save_ckpt {timing.save_ckpt:.2f}s | "
        f"total {timing.total:.2f}s"
    )

