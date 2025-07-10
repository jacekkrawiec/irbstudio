from abc import ABC, abstractmethod
from typing import Any, Dict 
from dataclasses import dataclass

@dataclass
class TestResult:
    test_name: str
    result: Dict[str, Any]
    passed: bool 
    message: str = ""

class BaseTest(ABC):
    """
    Abstract base class for credit risk models testing.
    Each tests implements run method, taking dataset and returning TestResult 
    """
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def run(self, dataset: Any) -> TestResult:
        """
        Runs test on dataset.
        Returns TestResult
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, params = {self.params})"