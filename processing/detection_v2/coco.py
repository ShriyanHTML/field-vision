"""
Class-id lookup for the COCO-pretrained RF-DETR stub.

RF-DETR's off-the-shelf checkpoints are trained on COCO, which only has a
generic "person" class — not separate player/goalkeeper/referee classes.
That split is reconstructed downstream by `team.TeamClassifier` from kit
colour until a fine-tuned checkpoint (with real per-role classes) replaces
this stub. See README.md.
"""

from rfdetr.assets.coco_classes import COCO_CLASSES

PERSON_CLASS_ID = 1
BALL_CLASS_ID = 37

# Fail loudly on an rfdetr version whose category ids have shifted, rather
# than silently detecting the wrong thing.
assert COCO_CLASSES[PERSON_CLASS_ID] == "person", (
    f"Expected COCO id {PERSON_CLASS_ID} to be 'person', "
    f"got {COCO_CLASSES.get(PERSON_CLASS_ID)!r} — rfdetr version mismatch?"
)
assert COCO_CLASSES[BALL_CLASS_ID] == "sports ball", (
    f"Expected COCO id {BALL_CLASS_ID} to be 'sports ball', "
    f"got {COCO_CLASSES.get(BALL_CLASS_ID)!r} — rfdetr version mismatch?"
)
