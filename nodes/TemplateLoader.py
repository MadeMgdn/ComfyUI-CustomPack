import os
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

NODE_CLASS_MAPPINGS_TEMPLATELOADER = {
    "CP PromptCompile": CP_PromptCompile,
    "CP PromptCompileEnhanced": CP_PromptCompileEnhanced,
}

NODE_DISPLAY_NAME_MAPPINGS_TEMPLATELOADER = {
    "CP PromptCompile": "Prompt compile",
    "CP PromptCompileEnhanced": "Prompt compile enhanced",
}