import time
from views import ScreenshotManager


if __name__ == "__main__":
    manager = ScreenshotManager(interval=1)

    try:
        manager.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
