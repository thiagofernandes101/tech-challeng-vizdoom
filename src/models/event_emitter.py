from abc import ABC, abstractmethod

from models.observer import Observer


class EventEmitter(ABC):

    @abstractmethod
    def subscribe(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def unsubscribe(self) -> None:
        pass

    @abstractmethod
    def notify(self) -> None:
        pass