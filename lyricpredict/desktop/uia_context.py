from __future__ import annotations

from typing import Any


UIA_TEXT_PATTERN_ID = 10014
UIA_VALUE_PATTERN_ID = 10002
UIA_CUIAUTOMATION_CLSID = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
TEXT_PATTERN_RANGE_ENDPOINT_START = 0
TEXT_PATTERN_RANGE_ENDPOINT_END = 1
TEXT_UNIT_CHARACTER = 0


def _first_range(selection: Any) -> Any | None:
    try:
        if len(selection) == 0:
            return None
        return selection[0]
    except TypeError:
        pass
    try:
        if selection.Length == 0:
            return None
        return selection.GetElement(0)
    except Exception:
        return None


def text_before_range(text_range: Any, context_window: int) -> str:
    current = text_range.Clone()
    try:
        current.MoveEndpointByRange(TEXT_PATTERN_RANGE_ENDPOINT_END, current, TEXT_PATTERN_RANGE_ENDPOINT_START)
    except Exception:
        pass
    current.MoveEndpointByUnit(TEXT_PATTERN_RANGE_ENDPOINT_START, TEXT_UNIT_CHARACTER, -int(context_window))
    text = current.GetText(int(context_window) + 1) or ""
    return text.strip()[-int(context_window) :]


def _exc_reason(prefix: str, exc: Exception) -> str:
    detail = str(exc).replace("\n", " ").strip()
    if len(detail) > 120:
        detail = detail[:117] + "..."
    return f"{prefix}:{type(exc).__name__}:{detail}" if detail else f"{prefix}:{type(exc).__name__}"


def _pattern_as(pattern: Any, interface_name: str) -> Any:
    if pattern is None:
        return None
    import comtypes.client

    comtypes.client.GetModule("UIAutomationCore.dll")
    import comtypes.gen.UIAutomationClient as uia

    interface = getattr(uia, interface_name)
    try:
        return pattern.QueryInterface(interface)
    except AttributeError:
        return pattern


def _create_comtypes_automation() -> tuple[Any | None, str]:
    try:
        import comtypes
        import comtypes.client
    except ImportError:
        return None, "comtypes_missing"
    try:
        comtypes.CoInitialize()
    except Exception:
        pass
    try:
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen.UIAutomationClient import CUIAutomation, IUIAutomation

        return comtypes.client.CreateObject(CUIAutomation, interface=IUIAutomation), "comtypes_interface"
    except Exception as interface_exc:
        interface_reason = _exc_reason("interface", interface_exc)
    try:
        return comtypes.client.CreateObject("UIAutomationClient.CUIAutomation"), "comtypes_progid"
    except Exception as progid_exc:
        try:
            return comtypes.client.CreateObject(UIA_CUIAUTOMATION_CLSID), "comtypes_clsid"
        except Exception as clsid_exc:
            return None, f"{interface_reason};{_exc_reason('progid', progid_exc)};{_exc_reason('clsid', clsid_exc)}"


def _read_from_comtypes(context_window: int) -> tuple[str, str]:
    automation, automation_reason = _create_comtypes_automation()
    if automation is None:
        return "", automation_reason
    try:
        focused = automation.GetFocusedElement()
        if focused is None:
            return "", "no_focused_element"
        try:
            pattern = _pattern_as(focused.GetCurrentPattern(UIA_TEXT_PATTERN_ID), "IUIAutomationTextPattern")
            if pattern is not None:
                selection = pattern.GetSelection()
                selected_range = _first_range(selection)
                if selected_range is not None:
                    text = text_before_range(selected_range, context_window)
                    if text:
                        return text, "text_pattern"
        except Exception as exc:
            text_pattern_reason = _exc_reason("text_pattern_error", exc)
        else:
            text_pattern_reason = "text_pattern_empty"
        try:
            value_pattern = _pattern_as(focused.GetCurrentPattern(UIA_VALUE_PATTERN_ID), "IUIAutomationValuePattern")
            value = getattr(value_pattern, "CurrentValue", "") if value_pattern is not None else ""
            value = str(value or "").strip()
            if value:
                return value[-int(context_window) :], "value_pattern"
        except Exception as exc:
            return "", f"{text_pattern_reason};{_exc_reason('value_pattern_error', exc)}"
        return "", f"{text_pattern_reason};value_pattern_empty"
    except Exception as exc:
        return "", _exc_reason(f"uia_error:{automation_reason}", exc)


def _read_from_pywinauto(context_window: int) -> tuple[str, str]:
    try:
        from pywinauto.uia_defines import IUIA
    except ImportError:
        return "", "pywinauto_missing"
    try:
        control = IUIA().get_focused_element()
        texts: list[str] = []
        try:
            value_pattern = _pattern_as(control.GetCurrentPattern(UIA_VALUE_PATTERN_ID), "IUIAutomationValuePattern")
            value = getattr(value_pattern, "CurrentValue", "") if value_pattern is not None else ""
            if value:
                texts.append(str(value))
        except Exception:
            pass
        try:
            text_pattern = _pattern_as(control.GetCurrentPattern(UIA_TEXT_PATTERN_ID), "IUIAutomationTextPattern")
            selection = text_pattern.GetSelection() if text_pattern is not None else None
            selected_range = _first_range(selection) if selection is not None else None
            if selected_range is not None:
                text = text_before_range(selected_range, context_window)
                if text:
                    texts.append(text)
        except Exception:
            pass
        for text in texts:
            text = text.strip()
            if text:
                return text[-int(context_window) :], "pywinauto"
        return "", "pywinauto_empty"
    except Exception as exc:
        return "", _exc_reason("pywinauto_error", exc)


def read_uia_context_with_reason(context_window: int) -> tuple[str, str]:
    context, reason = _read_from_comtypes(context_window)
    if context:
        return context, reason
    fallback_context, fallback_reason = _read_from_pywinauto(context_window)
    if fallback_context:
        return fallback_context, fallback_reason
    return "", f"{reason};{fallback_reason}"


def read_uia_context(context_window: int) -> str:
    context, _reason = read_uia_context_with_reason(context_window)
    return context
