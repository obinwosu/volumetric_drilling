#!/usr/bin/env python3
"""QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_scrollable_tabs.py"""
import pathlib
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scrollable_tabs as st  # noqa: E402


def wheel(widget, delta):
    point = QtCore.QPointF(widget.rect().center())
    return QtGui.QWheelEvent(point, QtCore.QPointF(widget.mapToGlobal(widget.rect().center())), QtCore.QPoint(),
                             QtCore.QPoint(0, delta), QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                             QtCore.Qt.NoScrollPhase, False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    root = QtWidgets.QVBoxLayout(window)
    tabs = QtWidgets.QTabWidget()
    root.addWidget(tabs)
    tall = QtWidgets.QWidget()
    tall.setObjectName("tall")
    layout = QtWidgets.QVBoxLayout(tall)
    combo = QtWidgets.QComboBox()
    combo.addItems(["a", "b", "c"])
    spacer = QtWidgets.QLabel("tall")
    spacer.setMinimumHeight(1500)
    start = QtWidgets.QPushButton("Start")
    for widget in (combo, spacer, start):
        layout.addWidget(widget)
    tabs.addTab(tall, "Tall")
    short = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(short).addWidget(QtWidgets.QLabel("short"))
    tabs.addTab(short, "Short")

    areas = st.make_tabs_scrollable(tabs, {tall: [start]})
    assert len(areas) == 2 and tabs.widget(0) is tall and tabs.tabText(1) == "Short"
    assert window.findChild(QtWidgets.QPushButton) is start and combo.window() is window

    window.resize(700, 500)
    window.show()
    app.processEvents()
    assert window.height() == 500 and window.minimumSizeHint().height() < 400
    bar = areas[0].verticalScrollBar()
    assert bar.maximum() > 1000                                 # the tall content scrolls
    assert start.parentWidget() is tall and start.isVisible()   # Start stays out of the scroll area
    assert start.geometry().bottom() <= tall.height()

    # The wheel over a dropdown scrolls the page and leaves its value alone.
    combo.setCurrentIndex(1)
    bar.setValue(200)
    QtWidgets.QApplication.sendEvent(combo, wheel(combo, -120))
    assert combo.currentIndex() == 1 and bar.value() > 200

    rect = QtCore.QRect(0, 27, 1366, 741)
    x, y, width, height = st.fit_to_screen(rect)
    assert QtCore.QRect(x, y, width, height).intersected(rect) == QtCore.QRect(x, y, width, height)
    assert (width, height) == (1326, 661)
    assert st.fit_to_screen(QtCore.QRect(0, 0, 3840, 2160))[2:] == (1800, 1200)
    window.close()
    print("scrollable tabs: all checks passed")


if __name__ == "__main__":
    main()
