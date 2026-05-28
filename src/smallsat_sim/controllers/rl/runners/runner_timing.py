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
    setup: float = 0.0
    sync: float = 0.0
    reset: float = 0.0
    setup_perturb_reset: float = 0.0
    setup_disturb_reset: float = 0.0
    setup_task_wrench: float = 0.0
    setup_scenario: float = 0.0
    setup_random_disturbance: float = 0.0

    @property
    def other_rollout(self) -> float:
        accounted = self.setup + self.scan + self.sync + self.buffer + self.reset
        return max(0.0, self.rollout - accounted)


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
        f"(setup {timing.setup:.2f}s, scan {timing.scan:.2f}s, sync {timing.sync:.2f}s, "
        f"buffer {timing.buffer:.2f}s, reset {timing.reset:.2f}s, other {timing.other_rollout:.2f}s) | "
        f"setup_detail perturb_reset {timing.setup_perturb_reset:.2f}s, "
        f"disturb_reset {timing.setup_disturb_reset:.2f}s, "
        f"task_wrench {timing.setup_task_wrench:.2f}s, "
        f"scenario {timing.setup_scenario:.2f}s, "
        f"random_disturbance {timing.setup_random_disturbance:.2f}s | "
        f"update {timing.update:.2f}s | "
        f"logging {timing.logging:.2f}s | "
        f"eval {timing.eval:.2f}s | "
        f"save_ckpt {timing.save_ckpt:.2f}s | "
        f"total {timing.total:.2f}s"
    )
