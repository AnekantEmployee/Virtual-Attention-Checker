import time
import streamlit as st
from views import ScreenshotManager


def main():
    manager = ScreenshotManager(interval=1)

    try:
        manager.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
