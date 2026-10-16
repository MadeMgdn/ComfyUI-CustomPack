import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.RgthreeImageComparer";
const TARGET_NODE_NAME = "Image Comparer (rgthree)";
const COMPARER_MODES = ["Slide", "Click"];
const DEFAULT_NODE_WIDTH = 420;
const DEFAULT_NODE_HEIGHT = 340;
const SELECTOR_HEIGHT = 20;

function imageDataToUrl(data) {
    const filename = encodeURIComponent(data?.filename ?? "");
    const type = encodeURIComponent(data?.type ?? "temp");
    const subfolder = encodeURIComponent(data?.subfolder ?? "");
    const previewFormat = app.getPreviewFormatParam?.() ?? "";
    const rand = app.getRandParam?.() ?? "";

    return api.apiURL(`/view?filename=${filename}&type=${type}&subfolder=${subfolder}${previewFormat}${rand}`);
}

function markDirty(node, foreground = true, background = false) {
    if (typeof node?.setDirtyCanvas === "function") {
        node.setDirtyCanvas(foreground, background);
        return;
    }
    app.graph?.setDirtyCanvas?.(foreground, background);
}

function suppressDefaultPreview(node) {
    node.imgs = [];
}

function measureText(ctx, text) {
    return ctx.measureText(text).width;
}

function pointInBounds(pos, bounds) {
    return (
        pos[0] >= bounds[0] &&
        pos[0] <= bounds[0] + bounds[2] &&
        pos[1] >= bounds[1] &&
        pos[1] <= bounds[1] + bounds[3]
    );
}

function toSerializableImage(image) {
    if (!image) {
        return image;
    }

    const serializable = { ...image };
    delete serializable.img;
    return serializable;
}

function normalizeWidgetValue(value) {
    let images = [];

    if (Array.isArray(value)) {
        images = value;
    } else if (value?.images && Array.isArray(value.images)) {
        images = value.images;
    }

    return images.map((image, index) => {
        if (!image || typeof image === "string") {
            return {
                name: index === 0 ? "A" : "B",
                selected: true,
                url: image || "",
            };
        }

        return { ...image };
    });
}

class CPImageComparerWidget {
    constructor(name, node) {
        this.name = name;
        this.node = node;
        this.type = "custom";
        this.options = {};
        this.hitAreas = {};
        this.selected = [];
        this._value = { images: [] };
    }

    set value(value) {
        let images = normalizeWidgetValue(value);

        if (images.length > 2) {
            const hasAAndB = images.some((image) => image.name?.startsWith("A")) &&
                images.some((image) => image.name?.startsWith("B"));

            if (!hasAAndB) {
                images = [images[0], images[1]].filter(Boolean);
            }
        }

        let selected = images.filter((image) => image.selected);
        if (!selected.length && images.length) {
            images[0].selected = true;
        }

        selected = images.filter((image) => image.selected);
        if (selected.length === 1 && images.length > 1) {
            const nextImage = images.find((image) => !image.selected);
            if (nextImage) {
                nextImage.selected = true;
            }
        }

        this._value.images = images;
        this.setSelected(images.filter((image) => image.selected));
    }

    get value() {
        return this._value;
    }

    setSelected(selected) {
        const nextSelected = selected.filter(Boolean).slice(0, 2);

        this._value.images.forEach((image) => {
            image.selected = false;
        });

        for (const image of nextSelected) {
            if (!image.img && image.url) {
                image.img = new Image();
                image.img.onload = () => markDirty(this.node, true, true);
                image.img.onerror = () => markDirty(this.node, true, true);
                image.img.src = image.url;
            }

            image.selected = true;
        }

        this.selected = nextSelected;
        this.node.imageIndex = Math.min(this.node.imageIndex ?? 0, Math.max(this.selected.length - 1, 0));
        markDirty(this.node, true, false);
    }

    draw(ctx, node, width, y) {
        this.hitAreas = {};
        let imageY = y;

        if (this.value.images.length > 2) {
            ctx.save();
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.font = "14px Arial";

            const drawData = [];
            const spacing = 6;
            let totalWidth = 0;

            for (const image of this.value.images) {
                const textWidth = measureText(ctx, image.name);
                drawData.push({
                    image,
                    text: image.name,
                    width: textWidth,
                });
                totalWidth += textWidth + spacing;
            }
            totalWidth = Math.max(0, totalWidth - spacing);

            let x = Math.max(0, (node.size[0] - totalWidth) / 2);
            for (const item of drawData) {
                ctx.fillStyle = item.image.selected ? "rgba(220, 220, 220, 1)" : "rgba(180, 180, 180, 0.55)";
                ctx.fillText(item.text, x, imageY);
                this.hitAreas[item.text] = {
                    bounds: [x, imageY, item.width, 16],
                    image: item.image,
                };
                x += item.width + spacing;
            }

            ctx.restore();
            imageY += SELECTOR_HEIGHT;
        }

        if (node.properties?.comparer_mode === "Click") {
            this.drawImage(ctx, this.selected[node.__cpImageComparerPointerDown ? 1 : 0] ?? this.selected[0], imageY);
            return;
        }

        this.drawImage(ctx, this.selected[0], imageY);
        if (node.__cpImageComparerPointerOver) {
            this.drawImage(ctx, this.selected[1], imageY, node.__cpImageComparerPointerPos?.[0]);
        }
    }

    drawImage(ctx, image, y, cropX = null) {
        const img = image?.img;
        if (!img?.naturalWidth || !img?.naturalHeight) {
            return;
        }

        const nodeWidth = Math.max(1, this.node.size[0]);
        const availableHeight = Math.max(1, this.node.size[1] - y - 4);
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const widgetAspect = nodeWidth / availableHeight;

        let targetWidth;
        let targetHeight;
        if (imageAspect > widgetAspect) {
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        } else {
            targetHeight = availableHeight;
            targetWidth = availableHeight * imageAspect;
        }

        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (availableHeight - targetHeight) / 2;
        let destWidth = targetWidth;
        let sourceWidth = img.naturalWidth;

        if (cropX !== null && cropX !== undefined) {
            const clippedX = Math.max(destX, Math.min(destX + targetWidth, cropX));
            destWidth = Math.max(0, clippedX - destX);
            sourceWidth = destWidth * (img.naturalWidth / targetWidth);
        }

        if (destWidth <= 0 || sourceWidth <= 0) {
            return;
        }

        ctx.save();
        ctx.beginPath();

        if (cropX !== null && cropX !== undefined) {
            ctx.rect(destX, destY, destWidth, targetHeight);
            ctx.clip();
        }

        ctx.drawImage(
            img,
            0,
            0,
            sourceWidth,
            img.naturalHeight,
            destX,
            destY,
            destWidth,
            targetHeight,
        );

        if (cropX !== null && cropX !== undefined && cropX >= destX && cropX <= destX + targetWidth) {
            const oldComposite = ctx.globalCompositeOperation;
            ctx.beginPath();
            ctx.moveTo(cropX, destY);
            ctx.lineTo(cropX, destY + targetHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255, 255, 255, 1)";
            ctx.stroke();
            ctx.globalCompositeOperation = oldComposite;
        }

        ctx.restore();
    }

    handlePointerDown(event, pos, node) {
        for (const hitArea of Object.values(this.hitAreas)) {
            if (!pointInBounds(pos, hitArea.bounds)) {
                continue;
            }

            const selected = [...this.selected];
            if (hitArea.image.name?.startsWith("A")) {
                selected[0] = hitArea.image;
            } else if (hitArea.image.name?.startsWith("B")) {
                selected[1] = hitArea.image;
            } else {
                selected[0] = hitArea.image;
            }

            this.setSelected(selected);
            event?.preventDefault?.();
            return true;
        }

        return false;
    }

    computeSize(width) {
        return [width, Math.max(260, DEFAULT_NODE_HEIGHT - 80)];
    }

    serializeValue() {
        return {
            images: this._value.images.map(toSerializableImage),
        };
    }
}

function ensureComparerModeProperty(node) {
    node.properties = node.properties || {};
    if (!COMPARER_MODES.includes(node.properties.comparer_mode)) {
        node.properties.comparer_mode = "Slide";
    }

    try {
        node.addProperty?.("comparer_mode", node.properties.comparer_mode, "combo", {
            values: COMPARER_MODES,
        });
    } catch (_error) {
        // Some ComfyUI builds reject duplicate properties; the value above is enough.
    }
}

function ensureWidget(node) {
    if (node.__cpImageComparerWidget) {
        return node.__cpImageComparerWidget;
    }

    ensureComparerModeProperty(node);
    node.serialize_widgets = true;
    node.imageIndex = node.imageIndex ?? 0;
    suppressDefaultPreview(node);
    node.__cpImageComparerPointerDown = false;
    node.__cpImageComparerPointerOver = false;
    node.__cpImageComparerPointerPos = [0, 0];

    const widget = new CPImageComparerWidget("rgthree_comparer", node);
    node.__cpImageComparerWidget = node.addCustomWidget(widget);

    const width = Math.max(DEFAULT_NODE_WIDTH, node.size?.[0] || 0);
    const height = Math.max(DEFAULT_NODE_HEIGHT, node.size?.[1] || 0);
    node.setSize?.([width, height]);
    markDirty(node, true, true);

    return node.__cpImageComparerWidget;
}

function setPointerDown(node, down = node.__cpImageComparerPointerDown) {
    const isDown = down && !!app.canvas?.pointer_is_down;

    if (node.__cpImageComparerPointerDown !== isDown) {
        node.__cpImageComparerPointerDown = isDown;
        markDirty(node, true, false);
    }

    node.imageIndex = isDown ? 1 : 0;

    if (isDown) {
        requestAnimationFrame(() => setPointerDown(node));
    }
}

function outputToWidgetValue(output) {
    if ("images" in output) {
        return {
            images: (output.images || []).map((data, index) => ({
                name: index === 0 ? "A" : "B",
                selected: true,
                url: imageDataToUrl(data),
            })),
        };
    }

    const aImages = output.a_images || [];
    const bImages = output.b_images || [];
    const images = [];
    const multiple = aImages.length + bImages.length > 2;

    for (const [index, data] of aImages.entries()) {
        images.push({
            name: aImages.length > 1 || multiple ? `A${index + 1}` : "A",
            selected: index === 0,
            url: imageDataToUrl(data),
        });
    }

    for (const [index, data] of bImages.entries()) {
        images.push({
            name: bImages.length > 1 || multiple ? `B${index + 1}` : "B",
            selected: index === 0,
            url: imageDataToUrl(data),
        });
    }

    return { images };
}

app.registerExtension({
    name: EXTENSION_NAME,

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        nodeType["@comparer_mode"] = {
            type: "combo",
            values: COMPARER_MODES,
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            ensureWidget(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info, ...rest) {
            const widget = ensureWidget(this);
            const widgetIndex = this.widgets?.indexOf(widget) ?? -1;
            const result = originalConfigure?.apply(this, [info, ...rest]);

            if (widgetIndex >= 0 && info?.widgets_values?.[widgetIndex] !== undefined) {
                widget.value = info.widgets_values[widgetIndex];
            }

            ensureWidget(this);
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output, ...rest) {
            const result = originalOnExecuted?.apply(this, [output, ...rest]);
            suppressDefaultPreview(this);
            ensureWidget(this).value = outputToWidgetValue(output || {});
            return result;
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (serialized, ...rest) {
            const result = originalOnSerialize?.apply(this, [serialized, ...rest]);

            for (const [index, widgetValue] of (serialized?.widgets_values || []).entries()) {
                if (this.widgets?.[index]?.name === "rgthree_comparer") {
                    serialized.widgets_values[index] = this.widgets[index].serializeValue?.() ?? widgetValue;
                }
            }

            return result;
        };

        const originalOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, pos, canvas) {
            const widget = ensureWidget(this);
            if (widget.handlePointerDown(event, pos, this)) {
                return true;
            }

            const result = originalOnMouseDown?.apply(this, [event, pos, canvas]);
            setPointerDown(this, true);
            return result ?? false;
        };

        const originalOnMouseEnter = nodeType.prototype.onMouseEnter;
        nodeType.prototype.onMouseEnter = function (event, ...rest) {
            const result = originalOnMouseEnter?.apply(this, [event, ...rest]);
            this.__cpImageComparerPointerOver = true;
            setPointerDown(this, !!app.canvas?.pointer_is_down);
            markDirty(this, true, false);
            return result;
        };

        const originalOnMouseLeave = nodeType.prototype.onMouseLeave;
        nodeType.prototype.onMouseLeave = function (event, ...rest) {
            const result = originalOnMouseLeave?.apply(this, [event, ...rest]);
            this.__cpImageComparerPointerOver = false;
            setPointerDown(this, false);
            markDirty(this, true, false);
            return result;
        };

        const originalOnMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, pos, canvas) {
            const result = originalOnMouseMove?.apply(this, [event, pos, canvas]);
            this.__cpImageComparerPointerPos = [...pos];
            this.imageIndex = pos[0] > this.size[0] / 2 ? 1 : 0;
            markDirty(this, true, false);
            return result;
        };
    },
});
