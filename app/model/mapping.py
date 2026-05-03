"""knowledgeTag ↔ skillID 양방향 매핑."""

from __future__ import annotations

import os
from typing import Optional


class TagSkillMapping:
    def __init__(self) -> None:
        self._tag_to_skill: dict[int, int] = {}
        self._skill_to_tag: dict[int, int] = {}

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise RuntimeError(f"매핑 파일을 찾을 수 없습니다: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                try:
                    tag, skill = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                self._tag_to_skill[tag] = skill
                self._skill_to_tag[skill] = tag

        if not self._tag_to_skill:
            raise RuntimeError(f"매핑 파일이 비어 있습니다: {path}")

    def to_skill(self, tag: int) -> Optional[int]:
        return self._tag_to_skill.get(tag)

    def to_tag(self, skill: int) -> Optional[int]:
        return self._skill_to_tag.get(skill)

    def __len__(self) -> int:
        return len(self._tag_to_skill)


# 모듈 싱글톤 — main.py 의 lifespan 에서 load() 호출
mapping = TagSkillMapping()
