"""Smart Tagging as a list of intervals in video frames.

Times in the XML are seconds from the first frame of the tactical video, checked by the official
starter against both kickoffs. Team names map to the submission letters here so nobody downstream
has to remember that the file spells it 'Atletic Club'.
"""

import xml.etree.ElementTree as ET

from . import config

TEAM_LETTER = {"Barcelona": "H", "Atletic Club": "A"}


def load():
    events = []
    for inst in ET.parse(config.asset("smart_tagging")).iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        events.append({
            "id": int(inst.findtext("ID")),
            "code": inst.findtext("code"),
            "start": int(round(start_s * config.FPS)),
            "end": int(round(end_s * config.FPS)),
            "team": TEAM_LETTER.get(tags.get("Team")),   # None for N/A (all throw-ins)
            "type": tags.get("Type"),
            "half": tags.get("Half"),
            "tags": tags,
        })
    events.sort(key=lambda e: e["start"])
    return events


def active_at(events, frame):
    return [e for e in events if e["start"] <= frame <= e["end"]]
