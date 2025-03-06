class SystemLogger:
    def __init__(self):
        self._xs = []
        self._us = []


class ControllerLogger:
    def __init__(self):
        self._xtrajs = []
        self._utrajs = []

class HumanTrajectoryLogger:
    def __init__(self) -> None:
        self._states = []


class LocalPlannerLogger:
    def __init__(self):
        self._trajs = []


class GlobalPlannerLogger:
    def __init__(self):
        self._paths = []
