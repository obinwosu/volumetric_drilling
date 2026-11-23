"""Makes every launcher tab scroll, so the window always fits the screen.

A QTabWidget is at least as big as its largest page, so one tall tab used to push the
whole window, and Start Simulation, off the bottom of the screen. Each page keeps its
object and its widgets; its contents move into a scroll area inside it. Widgets listed
as pinned stay outside the scroll area, always visible.
"""
from PyQt5 import QtCore, QtWidgets

WHEEL_WIDGETS = (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox, QtWidgets.QSlider)


class WheelGuard(QtCore.QObject):
    """Scrolling the page over a dropdown or number box must not change its value."""

    def eventFilter(self, widget, event):
        if event.type() != QtCore.QEvent.Wheel or not isinstance(widget, WHEEL_WIDGETS):
            return False
        # A number box or slider being edited keeps the wheel. A dropdown never does.
        if widget.hasFocus() and not isinstance(widget, QtWidgets.QComboBox):
            return False
        parent = widget.parentWidget()
        while parent is not None and not isinstance(parent, QtWidgets.QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            bar = parent.verticalScrollBar()
            bar.setValue(bar.value() - int(event.angleDelta().y() / 120 * bar.singleStep() * 3))
        return True


def make_scrollable(page, pinned=(), guard=None):
    """Moves the page's contents into a scroll area. Returns the scroll area."""
    content = QtWidgets.QWidget()
    content.setObjectName(f"{page.objectName() or type(page).__name__}_content")
    # Moves the layout and its widgets to `content`.
    content.setLayout(page.layout())

    scroll = QtWidgets.QScrollArea(page)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.verticalScrollBar().setSingleStep(24)
    scroll.horizontalScrollBar().setSingleStep(24)

    outer = QtWidgets.QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(scroll, 1)
    for widget in pinned:
        content.layout().removeWidget(widget)
        outer.addWidget(widget)
    if guard is not None:
        for widget in content.findChildren(WHEEL_WIDGETS):
            widget.installEventFilter(guard)
    return scroll


def make_tabs_scrollable(tabs, pinned=None):
    """Every page of `tabs`. `pinned` maps a page to widgets kept out of its scroll area."""
    guard = WheelGuard(tabs)
    return [make_scrollable(tabs.widget(index), (pinned or {}).get(tabs.widget(index), ()), guard)
            for index in range(tabs.count())]


def fit_to_screen(available, max_width=1800, max_height=1200, margin_x=40, margin_y=80):
    """Window (x, y, width, height) inside a screen's available QRect."""
    width = min(max_width, max(1, available.width() - margin_x))
    height = min(max_height, max(1, available.height() - margin_y))
    x = available.x() + max(0, (available.width() - width) // 2)
    y = available.y() + max(0, (available.height() - height) // 2)
    return x, y, width, height
