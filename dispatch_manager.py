import asyncio
import time
from typing import Dict

def is_valid_location(location: str) -> bool:
    if not location or not location.strip():
        return False
    loc = location.strip().lower()
    invalid_patterns = [
        "unknown", "not provided", "not yet provided", "specified location",
        "n/a", "none", "location not provided", "caller location", "pending", "unverified"
    ]
    if loc in invalid_patterns:
        return False
    for p in invalid_patterns:
        if loc.startswith(p):
            return False
    return True

class DispatchItem:
    def __init__(self, service_type: str, location: str, details: str, eta_minutes: int):
        self.service_type = service_type.strip().capitalize()
        self.location = location
        self.details = details
        self.eta = eta_minutes  # in simulated minutes
        self.status = "Dispatched"
        self.dispatch_time = time.strftime("%H:%M:%S")

class DispatchManager:
    def __init__(self):
        self.dispatches: Dict[str, DispatchItem] = {}

    def dispatch(self, service_type: str, location: str, details: str, eta_minutes: int = 5):
        key = service_type.strip().lower()
        valid_loc = is_valid_location(location)

        # If already dispatched to the same location, append the new details instead of resetting ETA/status
        if key in self.dispatches:
            existing = self.dispatches[key]
            # If location was updated from invalid to valid, update location and start dispatch
            if not is_valid_location(existing.location) and valid_loc:
                existing.location = location
                existing.status = "Dispatched"
                existing.eta = eta_minutes
                if details not in existing.details:
                    existing.details = f"{existing.details} | UPDATE: {details}"
                return

            if not location or existing.location == location or (not valid_loc and not is_valid_location(existing.location)):
                if details not in existing.details:
                    existing.details = f"{existing.details} | UPDATE: {details}"
                return

        # If location is invalid, place dispatch on HOLD
        if not valid_loc:
            self.dispatches[key] = DispatchItem(
                service_type=service_type,
                location="Awaiting Location",
                details=f"[HOLD - Location Required] {details}",
                eta_minutes=0
            )
            self.dispatches[key].status = "Hold - Location Required"
            return

        self.dispatches[key] = DispatchItem(
            service_type=service_type,
            location=location,
            details=details,
            eta_minutes=eta_minutes
        )

    def get_active_dispatches_str(self) -> str:
        """Returns a string representation of active dispatches for the LLM system prompt."""
        if not self.dispatches:
            return "No emergency services dispatched yet. Please dispatch services (police, paramedics, fire) as soon as you have their location and nature of emergency."
            
        lines = []
        for key, item in self.dispatches.items():
            lines.append(
                f"- {item.service_type}: Status={item.status}, ETA={item.eta} min, Location='{item.location}', Details='{item.details}'"
            )
        return "\n".join(lines)

    async def start_simulation_loop(self, on_update_callback=None):
        """Asynchronously ticks down ETAs and updates statuses.
        
        1 simulated minute passes every 10 seconds of real time.
        """
        while True:
            await asyncio.sleep(10)
            updated = False
            for key, dispatch in list(self.dispatches.items()):
                # Do not tick down units that are on hold awaiting location
                if dispatch.status.startswith("Hold") or dispatch.eta == 0:
                    continue

                if dispatch.status == "Dispatched":
                    dispatch.status = "En Route"
                    updated = True
                elif dispatch.status == "En Route":
                    if dispatch.eta > 1:
                        dispatch.eta -= 1
                        updated = True
                    elif dispatch.eta == 1:
                        dispatch.eta = 0
                        dispatch.status = "Arrived"
                        updated = True
            if updated and on_update_callback:
                on_update_callback()
