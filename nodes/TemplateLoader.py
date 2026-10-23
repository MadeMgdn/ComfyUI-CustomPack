from dataclasses import dataclass
import os
import re

import torch

categoryFolder = "Custom pack/Template loader"

#Prompt Collector (trigger, template, promt)
class Shared:
    TEMPLATES = {
        "None": {
            "type": "None",
            "positive": "",
            "negative": ""
        },
        "KFT IL": {
            "type": "IL",
            "positive": "kftenhance, masterpiece, best quality, amazing quality, realistic, very aesthetic, absurdres",
            "negative": "anime, 2d, bad quality, worst quality, worst detail, sketch, censor, simple background, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, artistic error, username, scan, (bad anatomy:1.3), censored, bar_censor, simple background, old"
        },
        "Pony": {
            "type": "Pony",
            "positive": "score_9, score_8_up, score_7",
            "negative": "score_6, score_5, score_4"
        },
        "Yogi": {
            "type": "IL",
            "positive": "embedding:3_Illustrious\\0_Dwn\\Stable_Yogis_Illustrious_Positives",
            "negative": "embedding:3_Illustrious\\0_Dwn\\Stable_Yogis_Illustrious_Negatives-neg"
        },
        "FurryNova IL": {
            "type": "IL",
            "positive": "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, furry, anthro, BREAK, detailed background, detailed hair, depth of field, detailed fluffy fur, photorealistic details, volumetric lighting",
            "negative": "human, multiple tails, modern, recent, old, oldest, graphic, cartoon, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, very displeasing, (worst quality, bad quality:1.2), bad anatomy, sketch, jpeg artifacts, signature, watermark, username, simple background, conjoined, bad ai-generated"
        },
        "EVO IL": {
            "type": "IL",
            "positive": "masterpiece, best quality, ultra-detailed, 8k resolution, high dynamic range, absurdres, sharp focus",
            "negative": "(worst quality:2), (low quality:2), (normal quality:2), bad anatomy, bad proportions, poorly drawn face, poorly drawn hands, missing fingers, extra limbs, blurry, pixelated, distorted, lowres, jpeg artifacts, watermark, signature, text, (deformed:1.5), (bad hands:1.3), overexposed, underexposed, censored, mutated, extra fingers, cloned face, bad eyes"
        }
    }

class CP_PromptCompile:
    def __init__(self):
        pass
    
    #const
    p = "positive"
    n = "negative"

    @classmethod
    def INPUT_TYPES(cls):
        templates = list(Shared.TEMPLATES.keys())
        return {
            "required": {
                "trigger": ("STRING", {"default": "", "multiline": False}),
                "template": (templates, {"default": templates[0]}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = (p, n)
    FUNCTION = "compile_prompt"
    CATEGORY = categoryFolder

    def compile_prompt(self, trigger, template, positive, negative):
        #const
        t = "type"
        IL = "IL"
        PONY = "Pony"
        sep = ", "
        E = ""
        NONE = "None"

        #common
        selected = Shared.TEMPLATES[template]

        isETr = True if trigger != E else False
        isETe = True if selected[t] != NONE else False
        isEPP = True if positive != E else False
        isENP = True if negative != E else False

        #positive
        pos = E
        pos += trigger if isETr else E
        pos += sep if (isETr and isETe) else E
        pos += selected[self.p] if isETe else E
        pos += sep if (isETe and isEPP) or (isETr and not isETe and isEPP) else E
        pos += positive if isEPP else E

        #negative
        neg = E
        if selected[t] == PONY:
            neg += selected[self.n]
            neg += sep if isENP else E
            neg += negative if isENP else E
        elif selected[t] == IL:
            neg += negative if isENP else E
            neg += sep if isENP else E
            neg += selected[self.n]
        elif selected[t] == NONE:
            neg += negative
        else:
            neg += negative if isENP else E
            neg += sep if isENP else E
            neg += selected[self.n]
        return (pos, neg)

#Extended version of the Promt assembler with morphing and averaging (for creating hybrids), and output of final conditioning
class CP_PromptCompileEnhanced:
    def __init__(self):
        pass

    # const
    p = "positive"
    n = "negative"

    @classmethod
    def INPUT_TYPES(cls):
        templates = list(Shared.TEMPLATES.keys())
        return {
            "required": {
                "clip": ("CLIP",),
                "trigger": ("STRING", {"default": "", "multiline": False}),
                "template": (templates, {"default": templates[0]}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "morph_power": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "morphpos": ("STRING", {"default": "", "multiline": True}),
                "morphneg": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative", "pos", "neg")
    FUNCTION = "compile_prompt"
    CATEGORY = categoryFolder

    def compile_prompt(self, clip, trigger, template, positive, negative, morphpos, morphneg, morph_power):
        # const
        t = "type"
        IL = "IL"
        PONY = "Pony"
        sep = ", "
        E = ""
        NONE = "None"

        # common
        selected = Shared.TEMPLATES[template]
        isETr = True if trigger != E else False
        isETe = True if selected[t] != NONE else False
        isEPP = True if positive != E else False
        isENP = True if negative != E else False

        # positive string compilation
        pos = E
        pos += trigger if isETr else E
        pos += sep if (isETr and isETe) else E
        pos += selected[self.p] if isETe else E
        pos += sep if (isETe and isEPP) or (isETr and not isETe and isEPP) else E
        pos += positive if isEPP else E

        # negative string compilation
        neg = E
        if selected[t] == PONY:
            neg += selected[self.n]
            neg += sep if isENP else E
            neg += negative if isENP else E
        elif selected[t] == IL:
            neg += negative if isENP else E
            neg += sep if isENP else E
            neg += selected[self.n]
        elif selected[t] == NONE:
            neg += negative
        else:
            neg += negative if isENP else E
            neg += sep if isENP else E
            neg += selected[self.n]

        # morph pos string compilation (identical logic to positive)
        isEMorphPos = True if morphpos != E else False
        morph_pos_text = E
        morph_pos_text += trigger if isETr else E
        morph_pos_text += sep if (isETr and isETe) else E
        morph_pos_text += selected[self.p] if isETe else E
        morph_pos_text += sep if (isETe and isEMorphPos) or (isETr and not isETe and isEMorphPos) else E
        morph_pos_text += morphpos if isEMorphPos else E

        # morph neg string compilation (identical logic to negative)
        isEMorphNeg = True if morphneg != E else False
        morph_neg_text = E
        if selected[t] == PONY:
            morph_neg_text += selected[self.n]
            morph_neg_text += sep if isEMorphNeg else E
            morph_neg_text += morphneg if isEMorphNeg else E
        elif selected[t] == IL:
            morph_neg_text += morphneg if isEMorphNeg else E
            morph_neg_text += sep if isEMorphNeg else E
            morph_neg_text += selected[self.n]
        elif selected[t] == NONE:
            morph_neg_text += morphneg
        else:
            morph_neg_text += morphneg if isEMorphNeg else E
            morph_neg_text += sep if isEMorphNeg else E
            morph_neg_text += selected[self.n]

        # CLIP encoding
        tokens_pos = clip.tokenize(pos)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        conditioning_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip.tokenize(neg)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        conditioning_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        tokens_morph_pos = clip.tokenize(morph_pos_text)
        cond_morph_pos, pooled_morph_pos = clip.encode_from_tokens(tokens_morph_pos, return_pooled=True)
        conditioning_morph_pos = [[cond_morph_pos, {"pooled_output": pooled_morph_pos}]]

        tokens_morph_neg = clip.tokenize(morph_neg_text)
        cond_morph_neg, pooled_morph_neg = clip.encode_from_tokens(tokens_morph_neg, return_pooled=True)
        conditioning_morph_neg = [[cond_morph_neg, {"pooled_output": pooled_morph_neg}]]

        def average_conditioning(conditioning_from, conditioning_to, conditioning_to_strength):
            out = []
            
            if len(conditioning_from) > 1:
                print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")
            
            if len(conditioning_to) > 1:
                print("Warning: ConditioningAverage conditioning_to contains more than 1 cond, only the first one will actually be applied to conditioning_from.")
            
            cond_from = conditioning_from[0][0]
            pooled_from = conditioning_from[0][1].get("pooled_output", None)
            cond_to = conditioning_to[0][0]
            pooled_to = conditioning_to[0][1].get("pooled_output", None)
            
            if cond_from.shape != cond_to.shape:
                max_length = max(cond_from.shape[1], cond_to.shape[1])
                
                if cond_from.shape[1] < max_length:
                    padding_size = max_length - cond_from.shape[1]
                    padding = torch.zeros(cond_from.shape[0], padding_size, cond_from.shape[2], 
                                        device=cond_from.device, dtype=cond_from.dtype)
                    cond_from = torch.cat([cond_from, padding], dim=1)
                
                if cond_to.shape[1] < max_length:
                    padding_size = max_length - cond_to.shape[1]
                    padding = torch.zeros(cond_to.shape[0], padding_size, cond_to.shape[2], 
                                        device=cond_to.device, dtype=cond_to.dtype)
                    cond_to = torch.cat([cond_to, padding], dim=1)
            
            averaged_cond = cond_from * (1.0 - conditioning_to_strength) + cond_to * conditioning_to_strength
            averaged_pooled = None
            if pooled_from is not None and pooled_to is not None:
                averaged_pooled = pooled_from * (1.0 - conditioning_to_strength) + pooled_to * conditioning_to_strength
            elif pooled_from is not None:
                averaged_pooled = pooled_from
            elif pooled_to is not None:
                averaged_pooled = pooled_to
            out_dict = {}
            if averaged_pooled is not None:
                out_dict["pooled_output"] = averaged_pooled
            
            out.append([averaged_cond, out_dict])
            return out
        final_pos_conditioning = average_conditioning(conditioning_pos, conditioning_morph_pos, morph_power)
        final_neg_conditioning = average_conditioning(conditioning_neg, conditioning_morph_neg, morph_power)

        return (pos, neg, final_pos_conditioning, final_neg_conditioning)

TEMPLATE_PRESETS = {
    "Type1 — horizontal A-pose (compact)": {
        "source": "Type1.txt",
        "layout": "row",
        "pose": "a_pose",
        "detail": 1,
        "safe_composition": False,
        "reconstruction": False,
    },
    "Type2 — horizontal T-pose": {
        "source": "Type2.txt",
        "layout": "row",
        "pose": "t_pose",
        "detail": 2,
        "safe_composition": False,
        "reconstruction": False,
    },
    "Type3 — 2x2 T-pose + safe margins": {
        "source": "Type3.txt",
        "layout": "grid",
        "pose": "t_pose",
        "detail": 3,
        "safe_composition": True,
        "reconstruction": False,
    },
    "Type4 — 2x2 T-pose + 3D reconstruction": {
        "source": "Type4.txt",
        "layout": "grid",
        "pose": "t_pose",
        "detail": 4,
        "safe_composition": True,
        "reconstruction": True,
    },
}

TEMPLATE_OPTIONS = tuple(TEMPLATE_PRESETS)


AUTO = "Auto (from description)"

SUBJECT_OPTIONS = (
    AUTO,
    "Human / humanoid",
    "Anthropomorphic creature",
    "Quadruped / animal",
    "Winged creature",
    "Aquatic creature",
    "Object / prop",
)

UPPER_LIMB_OPTIONS = (
    AUTO,
    "Hands + arms",
    "Paws + forelimbs",
    "Hooves + forelegs",
    "Wings",
    "Fins / flippers",
    "Tentacles",
    "None",
)

LOWER_LIMB_OPTIONS = (
    AUTO,
    "Feet + legs",
    "Paws + hind legs",
    "Hooves + hind legs",
    "Talons + legs",
    "Fins / tail",
    "Tentacles",
    "None",
)

SURFACE_OPTIONS = (
    AUTO,
    "Hair + clothing",
    "Fur",
    "Feathers",
    "Scales",
    "Skin / smooth hide",
    "Shell / exoskeleton",
    "Hard-surface materials",
    "Minimal / unspecified",
)

SUBJECT_OVERRIDE = {
    "Human / humanoid": "humanoid",
    "Anthropomorphic creature": "anthropomorphic",
    "Quadruped / animal": "quadruped",
    "Winged creature": "winged",
    "Aquatic creature": "aquatic",
    "Object / prop": "object",
}

UPPER_OVERRIDE = {
    "Hands + arms": "hands",
    "Paws + forelimbs": "paws",
    "Hooves + forelegs": "hooves",
    "Wings": "wings",
    "Fins / flippers": "fins",
    "Tentacles": "tentacles",
    "None": "none",
}

LOWER_OVERRIDE = {
    "Feet + legs": "feet",
    "Paws + hind legs": "paws",
    "Hooves + hind legs": "hooves",
    "Talons + legs": "talons",
    "Fins / tail": "fins",
    "Tentacles": "tentacles",
    "None": "none",
}

SURFACE_OVERRIDE = {
    "Hair + clothing": "human",
    "Fur": "fur",
    "Feathers": "feathers",
    "Scales": "scales",
    "Skin / smooth hide": "skin",
    "Shell / exoskeleton": "shell",
    "Hard-surface materials": "hard_surface",
    "Minimal / unspecified": "minimal",
}


KEYWORDS = {
    "anthropomorphic": (
        "anthropomorphic", "anthro", "furry", "humanoid animal",
        "антропоморф", "фурри", "человекоподобн",
    ),
    "object": (
        "box", "crate", "container", "chest", "package", "parcel", "barrel",
        "bottle", "cup", "mug", "chair", "table", "sofa", "cabinet", "object",
        "prop", "weapon", "sword", "gun", "vehicle", "car", "truck", "machine",
        "device", "appliance", "building", "короб", "ящик", "контейнер", "сундук",
        "бочк", "бутыл", "чашк", "стул", "стол", "диван", "шкаф", "предмет",
        "объект", "оруж", "меч", "машин", "автомоб", "устройств", "здани",
    ),
    "aquatic": (
        "fish", "shark", "dolphin", "whale", "seal", "ray", "eel", "aquatic",
        "fin", "fins", "flipper", "flippers",
        "mermaid", "рыб", "акул", "дельфин", "кит", "тюлень", "скат", "угорь",
        "водн", "русал", "плавник", "ласта",
    ),
    "winged": (
        "bird", "eagle", "owl", "raven", "crow", "hawk", "falcon", "parrot",
        "winged", "wings", "gryphon", "griffin", "птиц", "орёл", "сова", "ворон",
        "сокол", "попуг", "крыл", "грифон",
    ),
    "quadruped": (
        "quadruped", "animal", "dog", "cat", "wolf", "fox", "lion", "tiger",
        "bear", "horse", "pony", "cow", "bull", "goat", "sheep", "deer", "boar",
        "rabbit", "hare", "mouse", "rat", "elephant", "giraffe", "zebra", "camel",
        "four-legged", "paw", "paws", "forelimb", "forelimbs", "hind leg", "hind legs",
        "четвероног", "лап", "животн", "собак", "кошк", "кот", "волк",
        "лис", "лев", "тигр", "медвед", "лошад", "пони", "коров", "бык", "коз",
        "овц", "олен", "кабан", "крол", "заяц", "мыш", "крыс", "слон", "жираф",
        "зебр", "верблюд",
    ),
    "humanoid": (
        "human", "humanoid", "person", "character", "woman", "man", "girl", "boy",
        "warrior", "soldier", "knight", "wizard", "android", "cyborg", "robot",
        "человек", "гуманоид", "персонаж", "женщин", "мужчин", "девуш", "парень",
        "воин", "солдат", "рыцарь", "маг", "андроид", "киборг", "робот",
    ),
    "mechanical": (
        "robot", "android", "cyborg", "mecha", "mechanical", "hard-surface",
        "робот", "андроид", "киборг", "меха", "механическ",
    ),
    "hooves": (
        "hoof", "hooves", "horse", "pony", "cow", "bull", "goat", "sheep", "deer",
        "копыт", "лошад", "пони", "коров", "бык", "коз", "овц", "олен",
    ),
    "scales": (
        "scales", "scaled", "reptile", "lizard", "snake", "dragon", "crocodile",
        "чешу", "рептил", "ящер", "зме", "дракон", "крокодил",
    ),
    "shell": (
        "shell", "exoskeleton", "insect", "beetle", "crab", "lobster", "armor plates",
        "панцир", "экзоскелет", "насеком", "жук", "краб", "омар", "хитин",
    ),
    "feathers": ("feather", "plumage", "перья", "оперени"),
    "fur": ("fur", "furry", "coat", "шерсть", "мех", "пушист"),
    "hair": (
        "hair", "hairstyle", "braid", "ponytail", "beard", "mustache",
        "волос", "причес", "причёс", "косич", "хвост", "бород", "ус",
    ),
    "bald": ("bald", "hairless", "лыс", "без волос"),
    "clothing": (
        "clothes", "clothing", "outfit", "dress", "shirt", "jacket", "coat", "pants",
        "trousers", "skirt", "robe", "uniform", "armor", "costume", "hoodie", "sweater",
        "одеж", "наряд", "плать", "рубаш", "куртк", "пальто", "брюк", "штан", "юбк",
        "манти", "форм", "брон", "костюм", "худи", "свитер",
    ),
    "no_clothing": ("nude", "naked", "unclothed", "без одежды", "обнажен", "голый"),
    "footwear": (
        "shoe", "boot", "sandal", "sneaker", "slipper", "обув", "ботин", "сапог",
        "сандал", "кроссов", "тапоч",
    ),
    "barefoot": ("barefoot", "босиком", "босой"),
    "accessories": (
        "accessory", "accessories", "belt", "bag", "backpack", "glasses", "goggles",
        "jewelry", "necklace", "bracelet", "watch", "hat", "helmet", "scarf", "collar",
        "аксессуар", "ремень", "сумк", "рюкзак", "очк", "украшен", "ожерель",
        "браслет", "часы", "шляп", "шлем", "шарф", "ошейник",
    ),
    "face": ("face", "eyes", "mouth", "facial", "лицо", "глаз", "рот", "морд"),
}


@dataclass(frozen=True)
class Context:
    subject: str
    upper: str
    lower: str
    surface: str
    has_face: bool
    has_hair: bool
    has_clothing: bool
    has_footwear: bool
    has_accessories: bool


def _normalise(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has(text, group):
    for token in KEYWORDS[group]:
        # English entries are matched as complete words/phrases so that, for
        # example, "gray" is not mistaken for the aquatic keyword "ray".
        if token.isascii():
            pattern = r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        # Russian entries intentionally use stems (кошк, механическ, etc.).
        elif token in text:
            return True
    return False


def _resolve_subject(text, override):
    if override != AUTO:
        return SUBJECT_OVERRIDE[override]
    if _has(text, "anthropomorphic"):
        return "anthropomorphic"
    if _has(text, "object") and not _has(text, "humanoid"):
        return "object"
    if _has(text, "aquatic"):
        return "aquatic"
    if _has(text, "winged"):
        return "winged"
    if _has(text, "quadruped"):
        return "quadruped"
    return "humanoid"


def _resolve_upper(text, subject, override):
    if override != AUTO:
        return UPPER_OVERRIDE[override]
    if subject == "object":
        return "none"
    if "tentacle" in text or "щупал" in text:
        return "tentacles"
    if subject == "aquatic":
        return "fins"
    if subject == "winged":
        return "wings"
    if subject == "quadruped":
        return "hooves" if _has(text, "hooves") else "paws"
    if subject == "anthropomorphic" and not ("human hands" in text or "человеческие руки" in text):
        return "paws"
    return "hands"


def _resolve_lower(text, subject, override):
    if override != AUTO:
        return LOWER_OVERRIDE[override]
    if subject == "object":
        return "none"
    if "tentacle" in text or "щупал" in text:
        return "tentacles"
    if subject == "aquatic":
        return "fins"
    if subject == "winged":
        return "talons"
    if subject == "quadruped":
        return "hooves" if _has(text, "hooves") else "paws"
    if subject == "anthropomorphic" and not ("human feet" in text or "человеческие ступни" in text):
        return "paws"
    return "feet"


def _resolve_surface(text, subject, override):
    if override != AUTO:
        return SURFACE_OVERRIDE[override]
    if subject == "object" or _has(text, "mechanical"):
        return "hard_surface"
    if _has(text, "shell"):
        return "shell"
    if _has(text, "feathers") or subject == "winged":
        return "feathers"
    if _has(text, "scales"):
        return "scales"
    if _has(text, "fur") or subject in ("quadruped", "anthropomorphic"):
        return "fur"
    if subject == "aquatic":
        return "skin"
    return "human"


def infer_context(description, subject_type, upper_limb_type, lower_limb_type, surface_type):
    text = _normalise(description)
    subject = _resolve_subject(text, subject_type)
    upper = _resolve_upper(text, subject, upper_limb_type)
    lower = _resolve_lower(text, subject, lower_limb_type)
    surface = _resolve_surface(text, subject, surface_type)
    mechanical = surface == "hard_surface" or _has(text, "mechanical")

    if subject == "object":
        has_face = _has(text, "face")
        has_hair = False
        has_clothing = False
        has_footwear = False
        has_accessories = _has(text, "accessories")
    elif subject == "humanoid":
        has_face = True
        has_hair = not mechanical and not _has(text, "bald")
        has_clothing = not mechanical and not _has(text, "no_clothing")
        has_footwear = has_clothing and not _has(text, "barefoot")
        has_accessories = _has(text, "accessories")
    elif subject == "anthropomorphic":
        has_face = True
        has_hair = _has(text, "hair") and not _has(text, "bald")
        has_clothing = _has(text, "clothing") and not _has(text, "no_clothing")
        has_footwear = _has(text, "footwear") and not _has(text, "barefoot")
        has_accessories = _has(text, "accessories")
    else:
        has_face = True
        has_hair = False
        has_clothing = _has(text, "clothing")
        has_footwear = _has(text, "footwear")
        has_accessories = _has(text, "accessories")

    return Context(
        subject=subject,
        upper=upper,
        lower=lower,
        surface=surface,
        has_face=has_face,
        has_hair=has_hair,
        has_clothing=has_clothing,
        has_footwear=has_footwear,
        has_accessories=has_accessories,
    )


def _entity_words(context):
    if context.subject == "object":
        return "object", "objects", "complete object", "object design"
    if context.subject in ("quadruped", "winged", "aquatic"):
        return "creature", "creatures", "complete creature", "creature design"
    return "character", "characters", "full-body character", "character design"


def _surface_phrase(surface):
    return {
        "human": "skin appearance and fabric materials",
        "fur": "fur length, fur pattern and fur colors",
        "feathers": "plumage structure, feather pattern and feather colors",
        "scales": "scale pattern, scale size and scale colors",
        "skin": "skin texture, markings and colors",
        "shell": "shell or exoskeleton construction, segmentation and colors",
        "hard_surface": "construction, panels, seams, bevels, materials and surface finish",
        "minimal": "surface materials and colors",
    }[surface]


def _upper_terms(kind):
    return {
        "hands": ("arms", "hands", "fingertips", "elbows", "wrists"),
        "paws": ("forelimbs", "forepaws", "claw tips", "forelimb joints", "forepaw joints"),
        "hooves": ("forelegs", "front hooves", "hoof tips", "foreleg joints", "fetlocks"),
        "wings": ("wings", "wing ends", "wingtips", "wing joints", "outer wing joints"),
        "fins": ("pectoral fins or flippers", "fin ends", "fin tips", "fin joints", "outer fin joints"),
        "tentacles": ("upper tentacles", "tentacle ends", "tentacle tips", "tentacle joints", "outer tentacle sections"),
        "none": ("", "", "", "", ""),
    }[kind]


def _lower_terms(kind):
    return {
        "feet": ("legs", "feet", "toes", "knees", "ankles", "soles"),
        "paws": ("hind legs", "hind paws", "claw tips", "hind-leg joints", "hocks", "paw pads"),
        "hooves": ("hind legs", "hind hooves", "hoof tips", "hind-leg joints", "fetlocks", "hoof soles"),
        "talons": ("legs", "taloned feet", "claw tips", "knees", "ankles", "foot pads"),
        "fins": ("rear fins and tail", "tail end", "fin tips", "rear fin joints", "tail base", "underside"),
        "tentacles": ("lower tentacles", "tentacle ends", "tentacle tips", "tentacle joints", "tentacle bases", "underside"),
        "none": ("", "", "", "", "", ""),
    }[kind]


def _layout_block(layout, singular, complete):
    if layout == "row":
        return f"""VIEW LAYOUT:

Four {complete} views are arranged horizontally from left to right in exactly this order:
1. direct front view,
2. exact left-side orthographic profile,
3. exact right-side orthographic profile,
4. direct back view.

Each view contains exactly one {singular}."""

    return f"""VIEW LAYOUT:

The canvas contains exactly four separate and isolated square composition areas:

TOP LEFT: one direct front view.
TOP RIGHT: one direct back view.
BOTTOM LEFT: one exact left-side orthographic profile.
BOTTOM RIGHT: one exact right-side orthographic profile.

Each quadrant contains exactly one {complete}. The four views are separate technical views, not multiple subjects sharing one scene."""


def _directions_block(singular):
    return f"""VIEW DIRECTIONS:

The TOP LEFT {singular} is viewed directly from the front with its centerline facing the camera.
The TOP RIGHT {singular} is viewed directly from behind; no front-facing details are visible.
The BOTTOM LEFT {singular} is an exact 90-degree left-side profile, not a three-quarter view.
The BOTTOM RIGHT {singular} is an exact 90-degree right-side profile, not a three-quarter view.

The {singular} remains stationary while only the orthographic camera rotates around its vertical axis."""


def _consistency_block(context, singular):
    if context.subject == "object":
        fields = [
            "overall identity", "shape", "dimensions", "proportions", "silhouette",
            "construction", "panels", "seams", "edges", "openings", "materials",
            "colors", "surface finish", "markings",
        ]
        if context.has_face:
            fields.extend(("decorative face", "eye and mouth placement"))
        if context.has_accessories:
            fields.append("attachments and accessories")
    else:
        fields = ["identity"]
        if context.has_face:
            fields.extend(("facial structure", "eye shape", "head shape"))
        if context.has_hair:
            fields.extend(("hairstyle", "hair length", "hair color"))
        fields.extend(("body proportions", "total height", _surface_phrase(context.surface)))
        if context.upper != "none":
            upper = _upper_terms(context.upper)
            fields.extend((f"{upper[0]} length", f"{upper[1]} size"))
        if context.lower != "none":
            lower = _lower_terms(context.lower)
            fields.extend((f"{lower[0]} length", f"{lower[1]} size"))
        if context.has_clothing:
            fields.extend(("clothing construction", "clothing materials", "clothing colors"))
        if context.has_footwear:
            fields.extend(("footwear construction", "footwear size"))
        if context.has_accessories:
            fields.append("accessories and their exact placement")

    joined = ", ".join(dict.fromkeys(fields))
    return f"""CONSISTENCY:

The exact same {singular} appears in every view. Maintain identical {joined}.
Do not redesign, simplify, add, remove or reposition any described component between views."""


def _biped_upper_pose(context, pose):
    upper = _upper_terms(context.upper)
    if context.upper == "none":
        return "The upper body remains upright and symmetrical. No upper limbs are invented."

    limb, end, tip, joint, distal_joint = upper
    if pose == "a_pose":
        angle = "slightly downward and outward from the torso in a symmetrical A-pose"
    else:
        angle = "horizontally outward at exactly 90 degrees to the torso in a strict symmetrical T-pose"

    if context.upper == "hands":
        articulation = "Elbows are straight, wrists neutral, palms face downward, and fingers are straight and held together."
    elif context.upper == "paws":
        articulation = "The forelimb joints are straight, the forepaws are neutral, and all digits or claws are aligned together."
    elif context.upper == "wings":
        articulation = "Both wings have identical extension, feather or membrane arrangement, and clean fully visible wingtips."
    elif context.upper == "fins":
        articulation = "Both fins or flippers have identical extension and clean fully visible outer tips."
    elif context.upper == "tentacles":
        articulation = "The corresponding tentacles extend symmetrically without curling, tangling or crossing."
    else:
        articulation = f"The {joint} and {distal_joint} remain straight and symmetrical."

    return f"""Both {limb} extend {angle}. {articulation}

In the FRONT and BACK views, the complete span from one {tip} to the opposite {tip} is fully visible and remains inside the view area.

In the exact LEFT and RIGHT profiles, the far {limb[:-1] if limb.endswith('s') else limb} is completely hidden directly behind the near one. Only one clean {limb[:-1] if limb.endswith('s') else limb} silhouette is visible. This overlap is intentional and geometrically correct. Do not reveal the far limb by changing the camera angle."""


def _biped_lower_pose(context, reconstruction):
    if context.lower == "none":
        return "No lower limbs are invented. The lower body or base remains centered and unchanged."

    limb, end, tip, joint, ankle, underside = _lower_terms(context.lower)
    if not reconstruction:
        return f"""The {limb} are straight and symmetrical. The {end} are fully visible in the front and back views and remain aligned at the same level."""

    return f"""The {limb} are straight and separated only laterally toward the subject's left and right sides. The stance is slightly wider than hip-width but no wider than shoulder-width, forming a subtle symmetrical inverted V.

In the direct FRONT and BACK views, a narrow strip of empty background separates the inner {limb}, {joint}, {ankle} and {end}. They must not touch or merge.

The {end} remain level, parallel and at exactly the same depth position. There is absolutely no front-to-back offset; neither side is placed forward or backward.

In the exact LEFT and RIGHT profiles, the far {limb[:-1] if limb.endswith('s') else limb} and {end[:-1] if end.endswith('s') else end} are naturally hidden directly behind the near ones. This overlap is intentional. Do not separate them in depth or rotate to a three-quarter view."""


def _pose_block(context, pose, reconstruction):
    if context.subject == "object":
        return """OBJECT ORIENTATION:

The object remains rigid and unchanged in all four views. It is centered on the same vertical axis and ground level while only the orthographic camera rotates. Do not invent a body, anatomical pose or unsupported components."""

    if context.subject == "quadruped":
        lower = _lower_terms(context.lower)
        return f"""NEUTRAL QUADRUPED MODELING POSE:

The creature stands naturally on all four {lower[1]} with a straight stable body, level shoulders and level pelvis. The left and right limbs are separated laterally in the front and back views and remain at identical depth positions.

In exact side profiles, the far forelimb and far hind limb are naturally hidden behind the near limbs. The overlap is intentional. Do not stagger the limbs, create a walking pose or rotate to a three-quarter view."""

    if context.subject == "aquatic":
        return """NEUTRAL AQUATIC MODELING POSE:

The creature holds a straight neutral horizontal swimming pose with an uncurled body and tail. Paired fins or flippers are extended symmetrically. The body, fins and tail keep exactly the same position while only the camera rotates. No swimming motion, body bend or three-quarter view."""

    heading = "STRICT A-POSE" if pose == "a_pose" else "STRICT T-POSE"
    upper_block = _biped_upper_pose(context, pose)
    lower_block = _biped_lower_pose(context, reconstruction)
    return f"""{heading}:

The subject holds exactly the same rigid, symmetrical modeling pose in all four views.

{upper_block}

NEUTRAL LOWER-BODY STANCE:

{lower_block}"""


def _safe_composition_block(context, singular, safe):
    if context.subject == "object":
        visible = "entire silhouette, outermost edges, corners, base and top"
        extremity = "outermost left and right edges"
    else:
        items = ["entire head", "outer body silhouette"]
        if context.upper != "none":
            upper = _upper_terms(context.upper)
            items.extend((upper[0], upper[1], upper[2]))
        if context.lower != "none":
            lower = _lower_terms(context.lower)
            items.extend((lower[0], lower[1], lower[2]))
        visible = ", ".join(items)
        extremity = "outermost visible limb tips"

    if not safe:
        return f"""COMPOSITION:

Every view shows the complete {singular} at exactly the same scale and camera height. Keep the {visible} fully inside the canvas with even spacing and no overlap between neighboring views."""

    return f"""SAFE COMPOSITION:

Each {singular} is centered inside its own square quadrant. All four views use exactly the same scale and camera height.

Scale all views uniformly so the complete silhouette fits comfortably inside every quadrant. Use approximately 70 percent of the quadrant width and 80 percent of its height. Do not maximize the subject's height or width.

Maintain at least 12 percent empty background between the {extremity} and the quadrant edges. Leave empty background above the top and below the lowest point.

Leave a wide empty gutter between all quadrants. The {visible} must remain completely inside the corresponding quadrant. No component may cross a boundary, touch another view or become shared with a neighboring view."""


def _alignment_block(context):
    if context.subject == "object":
        return """IDENTICAL ALIGNMENT:

Every view has the same object height, width scale, centerline, base level and camera height. Corresponding construction features align consistently. Only the camera direction changes."""

    landmarks = ["top of the head", "shoulder level", "torso length", "hip or pelvis level"]
    if context.upper != "none":
        landmarks.extend((_upper_terms(context.upper)[3], _upper_terms(context.upper)[2]))
    if context.lower != "none":
        lower = _lower_terms(context.lower)
        landmarks.extend((lower[3], lower[4], lower[1]))
    joined = ", ".join(landmarks)
    return f"""IDENTICAL ALIGNMENT:

Every view has exactly the same total height, body proportions, centerline and camera height. The {joined} align consistently across all views. The pose never changes; only the camera direction changes."""


def _rendering_block(design_label, reconstruction):
    purpose = " suitable for 3D reconstruction and seamless texture reference" if reconstruction else ""
    return f"""RENDERING:

Strict orthographic projection, zero perspective distortion and zero foreshortening. Flat neutral studio lighting with identical exposure in every view. Plain uniform light-gray background with no horizon line, floor texture, cast shadows or environmental objects.

Clean technical {design_label} presentation{purpose}."""


def _required_block(context, layout, singular, plural):
    order = (
        "front, left profile, right profile and back"
        if layout == "row"
        else "front, back, left profile and right profile in their specified quadrants"
    )
    negatives = [
        "no repeated view", "no duplicated or missing subject", "no additional pose",
        "no three-quarter view", "no perspective", "no cropped silhouette",
        "no overlapping neighboring views", "no merged subjects", "no captions",
        "no labels", "no visible panel borders",
    ]
    if context.subject != "object":
        negatives.extend(("no changed proportions", "no asymmetrical pose"))
        if context.upper != "none":
            negatives.extend(("no cropped limb tips", "no fused upper limbs"))
        if context.lower != "none":
            negatives.extend(("no fused lower limbs in the front or back view", "no staggered lower limbs"))
    else:
        negatives.extend(("no changed construction", "no invented components", "no organic deformation"))

    return f"""REQUIRED RESULT:

Exactly four complete views of the same {singular}: {order}.

{', '.join(negatives)}."""


def build_prompt(description, extra_instructions, preset, context):
    singular, plural, complete, design_label = _entity_words(context)
    layout = preset["layout"]
    pose = preset["pose"]
    reconstruction = preset["reconstruction"]

    if context.subject == "object":
        title = f"A technical orthographic four-view turnaround sheet of the exact same {singular}"
    else:
        title = f"A technical orthographic four-view {design_label} turnaround sheet of the exact same {singular}"
    if layout == "grid":
        title += ", arranged as a strict 2-by-2 grid"
    else:
        title += ", arranged as a single horizontal row"
    if reconstruction:
        title += " and optimized for 3D reconstruction and seamless texture reference"
    title += "."

    blocks = [
        title,
        f"SUBJECT DESCRIPTION:\n\n{description.strip()}",
        _layout_block(layout, singular, complete),
    ]
    if layout == "grid" and preset["detail"] >= 3:
        blocks.append(_directions_block(singular))
    blocks.extend((
        _consistency_block(context, singular),
        _pose_block(context, pose, reconstruction),
        _safe_composition_block(context, singular, preset["safe_composition"]),
        _alignment_block(context),
        _rendering_block(design_label, reconstruction),
        _required_block(context, layout, singular, plural),
    ))
    if extra_instructions and extra_instructions.strip():
        blocks.append("ADDITIONAL REQUIREMENTS:\n\n" + extra_instructions.strip())
    return "\n\n".join(blocks).strip()


class CP_PromtGen3DMultiview:
    """Builds a context-aware Krea 2 multi-view prompt without runtime file I/O."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (TEMPLATE_OPTIONS, {"default": TEMPLATE_OPTIONS[-1]}),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Detailed description of the character, creature or object.",
                        "dynamicPrompts": False,
                    },
                ),
                "subject_type": (SUBJECT_OPTIONS, {"default": AUTO}),
                "upper_limb_type": (UPPER_LIMB_OPTIONS, {"default": AUTO}),
                "lower_limb_type": (LOWER_LIMB_OPTIONS, {"default": AUTO}),
                "surface_type": (SURFACE_OPTIONS, {"default": AUTO}),
                "extra_instructions": (
                    "STRING",
                    {"multiline": True, "default": "", "dynamicPrompts": False},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "detected_context")
    FUNCTION = "generate"
    CATEGORY = categoryFolder
    DESCRIPTION = "Context-aware Krea 2 four-view prompt generator with embedded templates."

    def generate(
        self,
        template,
        description,
        subject_type,
        upper_limb_type,
        lower_limb_type,
        surface_type,
        extra_instructions,
    ):
        if not description or not description.strip():
            raise ValueError("Description must not be empty.")
        preset = TEMPLATE_PRESETS[template]
        context = infer_context(
            description,
            subject_type,
            upper_limb_type,
            lower_limb_type,
            surface_type,
        )
        prompt = build_prompt(description, extra_instructions, preset, context)
        summary = (
            f"template={template}; subject={context.subject}; upper={context.upper}; "
            f"lower={context.lower}; surface={context.surface}; face={context.has_face}; "
            f"hair={context.has_hair}; clothing={context.has_clothing}; "
            f"footwear={context.has_footwear}; accessories={context.has_accessories}"
        )
        return (prompt, summary)


NODE_CLASS_MAPPINGS_TEMPLATELOADER = {
    "CP PromptCompile": CP_PromptCompile,
    "CP PromptCompileEnhanced": CP_PromptCompileEnhanced,
    "PromtGen3DMultiview": CP_PromtGen3DMultiview,
}

NODE_DISPLAY_NAME_MAPPINGS_TEMPLATELOADER = {
    "CP PromptCompile": "Prompt compile",
    "CP PromptCompileEnhanced": "Prompt compile enhanced",
    "PromtGen3DMultiview": "3D Multiview Prompt Generator (Krea 2)",
}
