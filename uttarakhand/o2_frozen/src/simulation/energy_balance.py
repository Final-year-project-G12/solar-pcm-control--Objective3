"""
src/simulation/energy_balance.py
===================================
Phase 3/4 — cumulative energy bookkeeping and the Gate 1 conservation check.

E_collector + E_initial  =  E_load + E_loss + E_final + E_residual

Pump energy (E_pump) is tracked and reported SEPARATELY, not folded into
the thermal conservation check — this follows the Phase 3 spec's own
instruction that "pump power ... [is] reported separately from thermal
energy" (framework doc, Phase 3 bullet list). Folding electrical pump work
into a water/PCM thermal balance would manufacture a residual that has
nothing to do with a real conservation violation, so Objective 2's Gate 1
is a strictly thermal check; E_pump appears in every performance report
as its own line item instead.

All quantities in Joules internally; converted to kWh only when reporting.
"""

from dataclasses import dataclass, field


J_PER_KWH = 3.6e6


@dataclass
class EnergyAccumulator:
    E_collector_J: float = 0.0
    E_load_J: float = 0.0
    E_loss_J: float = 0.0
    E_pump_J: float = 0.0
    E_unmet_J: float = 0.0
    E_charge_J: float = 0.0     # cumulative positive Q_pcm (water -> PCM)
    E_discharge_J: float = 0.0  # cumulative negative Q_pcm (PCM -> water), stored as positive magnitude
    E_initial_J: float = 0.0
    E_final_J: float = 0.0
    n_steps: int = 0
    n_failed_steps: int = 0

    def add_step(self, q_collector_w, q_load_w, q_pcm_w, q_loss_w, q_pump_w,
                 q_unmet_w, dt_s):
        self.E_collector_J += q_collector_w * dt_s
        self.E_load_J += q_load_w * dt_s
        self.E_loss_J += q_loss_w * dt_s
        self.E_pump_J += q_pump_w * dt_s
        self.E_unmet_J += q_unmet_w * dt_s
        if q_pcm_w >= 0:
            self.E_charge_J += q_pcm_w * dt_s
        else:
            self.E_discharge_J += -q_pcm_w * dt_s
        self.n_steps += 1

    def residual_report(self) -> dict:
        lhs = self.E_collector_J + self.E_initial_J
        rhs = self.E_load_J + self.E_loss_J + self.E_final_J
        residual_J = lhs - rhs
        denom = max(abs(self.E_collector_J), 1.0)
        residual_pct = abs(residual_J) / denom * 100.0
        return {
            "E_collector_kWh": self.E_collector_J / J_PER_KWH,
            "E_load_kWh": self.E_load_J / J_PER_KWH,
            "E_loss_kWh": self.E_loss_J / J_PER_KWH,
            "E_pump_kWh": self.E_pump_J / J_PER_KWH,
            "E_unmet_kWh": self.E_unmet_J / J_PER_KWH,
            "E_charge_kWh": self.E_charge_J / J_PER_KWH,
            "E_discharge_kWh": self.E_discharge_J / J_PER_KWH,
            "E_initial_kWh": self.E_initial_J / J_PER_KWH,
            "E_final_kWh": self.E_final_J / J_PER_KWH,
            "residual_J": residual_J,
            "residual_pct_of_collector": residual_pct,
            "n_steps": self.n_steps,
            "n_failed_steps": self.n_failed_steps,
        }
