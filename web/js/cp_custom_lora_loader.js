import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.CustomModelLoaders";
const PREVIEW_WIDTH = 384;
const PREVIEW_HEIGHT = 384;
const NODE_CONFIGS = {
    CP_CustomLoraLoader: {
        modelType: "lora",
        widgetName: "lora_name",
    },
    CP_CustomLoadCheckpoint: {
        modelType: "checkpoint",
        widgetName: "ckpt_name",
    },
    CP_CustomLoadDiffusionModel: {
        modelType: "diffusion_model",
        widgetName: "unet_name",
    },
};

let sidecars = {};
let sidecarsPromises = {};
let stylesInjected = false;
let previewImage = null;

function encodePath(value) {
    return encodeURIComponent(value).replace(/[!'()*]/g, (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`);
}

function getNodeConfig(node) {
    if (!node?.comfyClass) {
        return null;
    }
    return NODE_CONFIGS[node.comfyClass] || null;
}

function injectStyles() {
    if (stylesInjected) {
        return;
    }

    stylesInjected = true;
    const style = document.createElement("style");
    style.textContent = `
.cpcustom-model-preview {
    position: absolute;
    left: 0;
    top: 0;
    display: block;
    width: auto;
    height: auto;
    max-width: min(${PREVIEW_WIDTH}px, 44vw);
    max-height: min(${PREVIEW_HEIGHT}px, 70vh);
    background: #111;
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
    z-index: 99999;
    pointer-events: none;
}
.cpcustom-model-has-preview::after {
    content: "*";
    opacity: 0.7;
    padding-left: 4px;
}
.cpcustom-model-description {
    box-sizing: border-box;
    width: 100%;
    min-height: 116px;
    max-height: 220px;
    resize: vertical;
    border: 1px solid #3a3f47;
    border-radius: 6px;
    background: #15171a;
    color: #d8dee9;
    padding: 7px 8px;
    font: 12px/1.35 "Segoe UI", Tahoma, sans-serif;
    white-space: pre-wrap;
    outline: none;
}
.cpcustom-model-description:focus {
    border-color: #5aa7ff;
}
`;
    document.head.appendChild(style);
}

async function loadSidecars(modelType, force = false) {
    if (!force && sidecarsPromises[modelType]) {
        return sidecarsPromises[modelType];
    }

    sidecarsPromises[modelType] = api.fetchApi(`/cp_custom_model_loader/${modelType}/sidecars`)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            sidecars[modelType] = data || {};
            return sidecars[modelType];
        })
        .catch((error) => {
            console.warn(`[CustomPack] Failed to load ${modelType} sidecars`, error);
            delete sidecarsPromises[modelType];
            sidecars[modelType] = {};
            return sidecars[modelType];
        });

    return sidecarsPromises[modelType];
}

function getPreviewImage() {
    if (!previewImage) {
        previewImage = document.createElement("img");
        previewImage.className = "cpcustom-model-preview";
    }
    return previewImage;
}

function removePreviewImage() {
    if (previewImage?.parentElement) {
        previewImage.remove();
    }
}

function getPreviewRenderSize(imageElement, bodyRect) {
    const maxWidth = Math.min(PREVIEW_WIDTH, Math.max(1, bodyRect.width * 0.44));
    const maxHeight = Math.min(PREVIEW_HEIGHT, Math.max(1, bodyRect.height * 0.7));
    const naturalWidth = imageElement.naturalWidth || maxWidth;
    const naturalHeight = imageElement.naturalHeight || maxHeight;
    const scale = Math.min(maxWidth / naturalWidth, maxHeight / naturalHeight, 1);

    return {
        width: Math.max(1, Math.round(naturalWidth * scale)),
        height: Math.max(1, Math.round(naturalHeight * scale)),
    };
}

function positionPreview(relativeToElement, imageElement) {
    const itemRect = relativeToElement.getBoundingClientRect();
    const bodyRect = document.body.getBoundingClientRect();
    let left = itemRect.right;
    let top = itemRect.top;
    const imageSize = getPreviewRenderSize(imageElement, bodyRect);
    const imageWidth = imageSize.width;
    const imageHeight = imageSize.height;
    imageElement.style.width = `${imageWidth}px`;
    imageElement.style.height = `${imageHeight}px`;

    if (left + imageWidth > bodyRect.width) {
        left = itemRect.left - imageWidth;
    }

    if (top + imageHeight > bodyRect.height) {
        top = bodyRect.height - imageHeight;
    }
    if (top < 0) {
        top = 0;
    }

    imageElement.style.left = `${Math.max(0, Math.round(left))}px`;
    imageElement.style.top = `${Math.max(0, Math.round(top))}px`;
}

function normalizeComboValue(value, widget) {
    let normalized = value ?? widget?.value ?? "";
    if (normalized && typeof normalized === "object" && "content" in normalized) {
        normalized = normalized.content;
    }
    return String(normalized || "");
}

async function updateDescription(node, config, modelName) {
    const textarea = node.__cpCustomModelDescription;
    if (!textarea) {
        return;
    }

    const name = normalizeComboValue(modelName, null);
    if (!name || name === "None") {
        textarea.value = "";
        return;
    }

    const requestId = {};
    node.__cpCustomModelDescriptionRequest = requestId;

    try {
        const response = await api.fetchApi(
            `/cp_custom_model_loader/${config.modelType}/description/${encodePath(name)}?t=${Date.now()}`
        );
        const text = response.ok ? await response.text() : "";
        if (node.__cpCustomModelDescriptionRequest === requestId) {
            textarea.value = text;
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (error) {
        if (node.__cpCustomModelDescriptionRequest === requestId) {
            textarea.value = "";
        }
        console.warn(`[CustomPack] Failed to load ${config.modelType} description`, error);
    }
}

function addDescriptionWidget(node) {
    if (node.__cpCustomModelDescription) {
        return node.__cpCustomModelDescription;
    }

    const textarea = document.createElement("textarea");
    textarea.className = "cpcustom-model-description";
    textarea.readOnly = true;
    textarea.spellcheck = false;
    textarea.placeholder = "No description file";
    textarea.addEventListener("keydown", (event) => event.stopPropagation());
    textarea.addEventListener("pointerdown", (event) => event.stopPropagation());

    const widget = node.addDOMWidget("model_description", "CP_MODEL_DESCRIPTION", textarea, {
        hideOnZoom: false,
        getMinHeight() {
            return 130;
        },
        getMaxHeight() {
            return 235;
        },
        getHeight() {
            return 145;
        },
    });
    widget.serialize = false;

    node.__cpCustomModelDescription = textarea;
    return textarea;
}

function patchModelWidget(node, config) {
    const widget = node.widgets?.find((item) => item.name === config.widgetName);
    if (!widget || widget.__cpCustomModelPatched) {
        return;
    }

    widget.__cpCustomModelPatched = true;
    const originalCallback = widget.callback;
    let previousValue;

    widget.callback = function (...args) {
        let result = originalCallback?.apply(this, args);
        let selected = normalizeComboValue(result ?? widget.value, widget);

        if (result && typeof result === "object" && "content" in result) {
            widget.value = selected;
            result = selected;
        }

        if (selected !== previousValue) {
            previousValue = selected;
            updateDescription(node, config, selected);
        }

        return result ?? widget.value;
    };

    setTimeout(() => {
        previousValue = normalizeComboValue(widget.value, widget);
        updateDescription(node, config, previousValue);
    }, 30);
}

async function enhanceMenu(menu, config) {
    await loadSidecars(config.modelType);

    const items = menu.querySelectorAll(".litemenu-entry");
    for (const item of items) {
        if (item.__cpCustomModelPreviewPatched) {
            continue;
        }

        const name = normalizeComboValue(item.getAttribute("data-value") || item.textContent, null).trim();
        if (!name || !sidecars[config.modelType]?.[name]?.preview) {
            continue;
        }

        item.__cpCustomModelPreviewPatched = true;
        item.classList.add("cpcustom-model-has-preview");
        item.addEventListener("mouseover", () => {
            const imageElement = getPreviewImage();
            const src = `/cp_custom_model_loader/${config.modelType}/preview/${encodePath(name)}?t=${Date.now()}`;
            imageElement.style.visibility = "hidden";
            imageElement.onload = () => {
                if (imageElement.parentElement) {
                    positionPreview(item, imageElement);
                    imageElement.style.visibility = "visible";
                }
            };
            imageElement.onerror = removePreviewImage;
            imageElement.src = src;
            document.body.appendChild(imageElement);
        }, { passive: true });
        item.addEventListener("mousemove", () => {
            if (previewImage?.parentElement) {
                positionPreview(item, previewImage);
            }
        }, { passive: true });
        item.addEventListener("mouseout", removePreviewImage, { passive: true });
        item.addEventListener("click", removePreviewImage, { passive: true });
    }
}

function refreshAllSidecars(force = false) {
    return Promise.all(
        Object.values(NODE_CONFIGS).map((config) => loadSidecars(config.modelType, force))
    );
}

app.registerExtension({
    name: EXTENSION_NAME,
    init() {
        injectStyles();
        refreshAllSidecars();

        const refreshComboInNodes = app.refreshComboInNodes;
        if (refreshComboInNodes && !app.__cpCustomModelRefreshPatched) {
            app.__cpCustomModelRefreshPatched = true;
            app.refreshComboInNodes = async function (...args) {
                const result = await refreshComboInNodes.apply(this, args);
                await refreshAllSidecars(true);
                return result;
            };
        }

        const observer = new MutationObserver((mutations) => {
            const node = app.canvas.current_node;
            const config = getNodeConfig(node);
            if (!config) {
                return;
            }

            for (const mutation of mutations) {
                for (const removed of mutation.removedNodes) {
                    if (removed.classList?.contains("litecontextmenu")) {
                        removePreviewImage();
                    }
                }

                for (const added of mutation.addedNodes) {
                    if (!added.classList?.contains("litecontextmenu")) {
                        continue;
                    }

                    const widget = app.canvas.getWidgetAtCursor?.();
                    if (widget?.name !== config.widgetName) {
                        continue;
                    }

                    requestAnimationFrame(() => {
                        if (added.querySelector(".comfy-context-menu-filter")) {
                            enhanceMenu(added, config);
                        }
                    });
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: false });
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData?.name];
        if (!config) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            injectStyles();
            addDescriptionWidget(this);
            patchModelWidget(this, config);
            return result;
        };
    },
});
