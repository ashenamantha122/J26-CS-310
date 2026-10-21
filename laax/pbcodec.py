"""
Protocol Buffers codec for the LAAX contracts (proposal FR5).

laax/contracts.py stays the single definition every component codes against;
this module turns those dataclasses into the messages of proto/laax.proto and
back. The mapping is by field name, so a field added to a dataclass without a
matching proto field fails loudly in encode() (and in tests/test_transport.py)
instead of silently vanishing on the wire.

Two fields need a rule of their own:
    CorrelationAlert.path   a list of dicts -> path_json (JSON string)
    HostEvent.argv          None and [] both exist in the dataclass; proto3
                            repeated fields cannot tell them apart, so
                            argv_present carries the difference.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict

from . import laax_pb2
from .contracts import DECODERS, Message

_PB = {name: getattr(laax_pb2, name) for name in laax_pb2.DESCRIPTOR.message_types_by_name}


class CodecError(ValueError):
    pass


def _pb_class(msg_cls: type):
    try:
        return _PB[msg_cls.__name__]
    except KeyError:
        raise CodecError(f"no protobuf message for {msg_cls.__name__}") from None


_FIELDS: Dict[type, tuple] = {}


def _field_names(cls: type) -> tuple:
    """Dataclass field names, checked against the proto once per class (the
    encoder runs per telemetry event, so this is on the hot path)."""
    names = _FIELDS.get(cls)
    if names is None:
        pb_fields = _pb_class(cls).DESCRIPTOR.fields_by_name
        names = tuple(f.name for f in dataclasses.fields(cls))
        for name in names:
            wire = "path_json" if (cls.__name__ == "CorrelationAlert" and name == "path") else name
            if wire not in pb_fields:
                raise CodecError(f"{cls.__name__}.{name} has no proto field")
        _FIELDS[cls] = names
    return names


def encode(msg: Message) -> bytes:
    cls = type(msg)
    pb = _pb_class(cls)()
    for name in _field_names(cls):
        value = getattr(msg, name)
        if name == "path" and cls.__name__ == "CorrelationAlert":
            pb.path_json = json.dumps(value)
            continue
        if value is None:
            continue                       # optional: absence is the encoding
        if isinstance(value, list):
            getattr(pb, name).extend(value)
            if name == "argv":
                pb.argv_present = True
            continue
        try:
            setattr(pb, name, value)
        except (TypeError, ValueError) as exc:
            raise CodecError(f"{cls.__name__}.{name}={value!r}: {exc}") from exc
    return pb.SerializeToString()


def _has(pb, name: str) -> bool:
    """Presence for proto3 `optional` fields; plain fields are always present."""
    try:
        return pb.HasField(name)
    except ValueError:
        return True


def decode(topic: str, data: bytes) -> Message:
    try:
        cls = DECODERS[topic]
    except KeyError:
        raise CodecError(f"unknown topic {topic!r}") from None
    try:
        pb = _pb_class(cls).FromString(data)
    except Exception as exc:               # DecodeError: malformed or hostile frame
        raise CodecError(f"undecodable {cls.__name__} on {topic}: {exc}") from exc

    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name == "path" and cls.__name__ == "CorrelationAlert":
            kwargs["path"] = json.loads(pb.path_json) if pb.path_json else []
        elif f.name == "argv" and cls.__name__ == "HostEvent":
            kwargs["argv"] = list(pb.argv) if pb.argv_present else None
        else:
            fd = pb.DESCRIPTOR.fields_by_name[f.name]
            if fd.label == fd.LABEL_REPEATED:
                kwargs[f.name] = list(getattr(pb, f.name))
            elif _has(pb, f.name):
                kwargs[f.name] = getattr(pb, f.name)
            else:
                kwargs[f.name] = None
    return cls(**kwargs)
