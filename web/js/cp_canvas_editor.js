import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.CanvasLayerEditor";
const TARGET_NODE_NAME = "CP CanvasLayerEditor";

const TOOL = {
    MOVE: "move",
    SELECT_RECT: "select_rect",
    SELECT_LASSO: "select_lasso",
    TRANSFORM: "transform",
    BRUSH: "brush",
    ERASER: "eraser",
};

const BLEND_MODES = ["Normal", "Multiply", "Screen", "Overlay"];
const BLEND_TO_CANVAS = {
    Normal: "source-over",
    Multiply: "multiply",
    Screen: "screen",
    Overlay: "overlay",
};

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected) {
        return;
    }

    stylesInjected = true;
    const style = document.createElement("style");
    style.textContent = `
.cpce-root {
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    min-height: 940px;
    color: #dddddd;
    font: 12px/1.25 "Segoe UI", Tahoma, sans-serif;
}
.cpce-layout {
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    display: grid;
    grid-template-columns: 250px 1fr 300px;
    gap: 10px;
    padding: 8px;
    border-radius: 8px;
    background: #202326;
}
.cpce-panel {
    box-sizing: border-box;
    background: #181a1d;
    border: 1px solid #2b2f35;
    border-radius: 8px;
    overflow-y: auto;
    overflow-x: hidden;
}
.cpce-toolbar {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.cpce-toolbar-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
.cpce-btn {
    box-sizing: border-box;
    border: 1px solid #3a3f47;
    background: #2a2f36;
    color: #f0f0f0;
    border-radius: 6px;
    padding: 6px 8px;
    cursor: pointer;
}
.cpce-btn:hover {
    background: #343a43;
}
.cpce-btn.active {
    border-color: #5aa7ff;
    box-shadow: 0 0 0 1px #5aa7ff inset;
}
.cpce-btn:disabled {
    opacity: 0.55;
    cursor: default;
}
.cpce-section {
    box-sizing: border-box;
    border: 1px solid #2f3338;
    border-radius: 6px;
    padding: 7px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: #15171a;
}
.cpce-title {
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #f5f5f5;
}
.cpce-row {
    display: grid;
    grid-template-columns: 78px 1fr;
    gap: 6px;
    align-items: center;
}
.cpce-row.cpce-row-range {
    grid-template-columns: 78px minmax(0, 1fr) 58px;
}
.cpce-row input[type="range"] {
    width: 100%;
    min-width: 0;
}
.cpce-row input[type="number"] {
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
    background: #0f1113;
    border: 1px solid #3a3f47;
    border-radius: 4px;
    color: #efefef;
    padding: 3px 5px;
}
.cpce-row select,
.cpce-row input[type="color"] {
    width: 100%;
    box-sizing: border-box;
    background: #0f1113;
    border: 1px solid #3a3f47;
    border-radius: 4px;
    color: #efefef;
    padding: 3px 5px;
}
.cpce-center {
    box-sizing: border-box;
    border: 1px solid #2b2f35;
    border-radius: 8px;
    background: #141619;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    position: relative;
}
.cpce-canvas-box {
    position: relative;
    display: inline-block;
}
.cpce-canvas {
    display: block;
    image-rendering: auto;
    background: #0f1113;
}
.cpce-overlay {
    position: absolute;
    left: 0;
    top: 0;
    pointer-events: auto;
}
.cpce-empty {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: #b2b9c5;
    pointer-events: none;
}
.cpce-layers {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.cpce-layer-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 290px;
    overflow: auto;
}
.cpce-layer-item {
    border: 1px solid #2f3338;
    border-radius: 6px;
    background: #15171a;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.cpce-layer-item.selected {
    border-color: #5aa7ff;
    box-shadow: 0 0 0 1px #5aa7ff inset;
}
.cpce-layer-head {
    display: grid;
    grid-template-columns: 20px 1fr 24px;
    gap: 6px;
    align-items: center;
}
.cpce-layer-name {
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.cpce-layer-mini {
    display: grid;
    grid-template-columns: 46px minmax(0, 1fr) 56px;
    gap: 6px;
    align-items: center;
}
.cpce-layer-mini input[type="range"],
.cpce-layer-mini input[type="number"] {
    width: 100%;
    min-width: 0;
}
.cpce-tiny {
    font-size: 11px;
    opacity: 0.92;
}
.cpce-action-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
@media (max-width: 1280px) {
    .cpce-layout {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto auto;
        min-height: 1320px;
    }
    .cpce-center {
        min-height: 460px;
    }
}
`;
    document.head.appendChild(style);
}

function clamp(v, minV, maxV) {
    return Math.max(minV, Math.min(maxV, v));
}

function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

function rgbToHsv(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    const s = max === 0 ? 0 : d / max;
    const v = max;

    if (d !== 0) {
        switch (max) {
            case r:
                h = ((g - b) / d) % 6;
                break;
            case g:
                h = (b - r) / d + 2;
                break;
            default:
                h = (r - g) / d + 4;
                break;
        }
        h /= 6;
        if (h < 0) {
            h += 1;
        }
    }
    return [h, s, v];
}

function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0:
            return [v, t, p];
        case 1:
            return [q, v, p];
        case 2:
            return [p, v, t];
        case 3:
            return [p, q, v];
        case 4:
            return [t, p, v];
        default:
            return [v, p, q];
    }
}

class CanvasLayerEditor {
    constructor(node, rootElement) {
        this.node = node;
        this.root = rootElement;

        this.canvasWidth = 512;
        this.canvasHeight = 512;
        this.layers = [];
        this.selectedLayerId = null;
        this.tool = TOOL.MOVE;
        this.selection = null;

        this.interaction = null;
        this.pointerInside = false;

        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 100;
        this.historySuspended = false;

        this.loadingFromServer = false;
        this.saveTimer = null;
        this.stateVersion = 0;
        this.lastSavedVersion = -1;

        this.brush = {
            size: 40,
            hardness: 0.65,
            shape: "circle",
            color: "#ffffff",
        };
        this.brushStampCache = new Map();

        this.syncingControls = false;
        this.resizeObserver = null;
        this.apiHandlers = null;
        this.selectionTransform = null;

        this.ui = {};
        this.controls = {};

        this.buildUI();
        this.bindEvents();
        this.bindApiEvents();
        this.setTool(TOOL.MOVE);
        this.loadFromServer();
    }

    destroy() {
        this.unbindApiEvents();
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.saveTimer) {
            clearTimeout(this.saveTimer);
            this.saveTimer = null;
        }
    }

    buildUI() {
        this.root.className = "cpce-root";
        this.root.innerHTML = "";

        const layout = document.createElement("div");
        layout.className = "cpce-layout";

        const toolbar = document.createElement("div");
        toolbar.className = "cpce-panel cpce-toolbar";

        const toolGrid = document.createElement("div");
        toolGrid.className = "cpce-toolbar-grid";
        toolbar.appendChild(toolGrid);

        const addToolButton = (label, toolId, title = "", wide = false) => {
            const btn = document.createElement("button");
            btn.className = "cpce-btn";
            if (wide) {
                btn.style.gridColumn = "1 / -1";
            }
            btn.textContent = label;
            btn.title = title;
            btn.addEventListener("click", () => this.setTool(toolId));
            toolGrid.appendChild(btn);
            this.ui[`tool_${toolId}`] = btn;
        };

        addToolButton("Move", TOOL.MOVE, "Move selected layer", true);
        addToolButton("Rect Sel", TOOL.SELECT_RECT, "Rectangular selection");
        addToolButton("Lasso", TOOL.SELECT_LASSO, "Lasso selection");
        addToolButton("Brush", TOOL.BRUSH, "Brush tool");
        addToolButton("Eraser", TOOL.ERASER, "Eraser tool");

        const historyRow = document.createElement("div");
        historyRow.className = "cpce-action-row";
        const undoBtn = document.createElement("button");
        undoBtn.className = "cpce-btn";
        undoBtn.textContent = "Undo";
        undoBtn.addEventListener("click", () => this.undo());
        const redoBtn = document.createElement("button");
        redoBtn.className = "cpce-btn";
        redoBtn.textContent = "Redo";
        redoBtn.addEventListener("click", () => this.redo());
        historyRow.appendChild(undoBtn);
        historyRow.appendChild(redoBtn);
        toolbar.appendChild(historyRow);
        this.ui.undoBtn = undoBtn;
        this.ui.redoBtn = redoBtn;

        const selectionActions = document.createElement("div");
        selectionActions.className = "cpce-action-row";
        const deleteSelBtn = document.createElement("button");
        deleteSelBtn.className = "cpce-btn";
        deleteSelBtn.textContent = "Delete Sel";
        deleteSelBtn.addEventListener("click", () => this.deleteSelection());
        const clearSelBtn = document.createElement("button");
        clearSelBtn.className = "cpce-btn";
        clearSelBtn.textContent = "Clear Sel";
        clearSelBtn.addEventListener("click", () => {
            if (!this.selection) {
                return;
            }
            this.selection = null;
            this.selectionTransform = null;
            this.syncSelectedControls();
            this.render();
            this.commitStateChange();
        });
        selectionActions.appendChild(deleteSelBtn);
        selectionActions.appendChild(clearSelBtn);
        toolbar.appendChild(selectionActions);

        const reloadBtn = document.createElement("button");
        reloadBtn.className = "cpce-btn";
        reloadBtn.textContent = "Reload Inputs";
        reloadBtn.title = "Discard current editor state and reload images from latest Run";
        reloadBtn.addEventListener("click", () => this.confirmReloadInputs());
        toolbar.appendChild(reloadBtn);
        this.ui.reloadBtn = reloadBtn;

        const brushSection = document.createElement("div");
        brushSection.className = "cpce-section";
        brushSection.innerHTML = `<div class="cpce-title">Brush/Eraser</div>`;
        const brushSize = this.createRangeRow("Size", 1, 220, 1, this.brush.size, (value) => {
            this.brush.size = value;
        });
        const brushHardness = this.createRangeRow("Hard", 0.01, 1, 0.01, this.brush.hardness, (value) => {
            this.brush.hardness = value;
        });
        const brushShapeRow = document.createElement("div");
        brushShapeRow.className = "cpce-row";
        const shapeLabel = document.createElement("div");
        shapeLabel.textContent = "Shape";
        const shapeSelect = document.createElement("select");
        const optCircle = document.createElement("option");
        optCircle.value = "circle";
        optCircle.textContent = "Circle";
        const optSquare = document.createElement("option");
        optSquare.value = "square";
        optSquare.textContent = "Square";
        shapeSelect.appendChild(optCircle);
        shapeSelect.appendChild(optSquare);
        shapeSelect.value = this.brush.shape;
        shapeSelect.addEventListener("change", () => {
            this.brush.shape = shapeSelect.value === "square" ? "square" : "circle";
        });
        brushShapeRow.appendChild(shapeLabel);
        brushShapeRow.appendChild(shapeSelect);
        const brushColorRow = document.createElement("div");
        brushColorRow.className = "cpce-row";
        const colorLabel = document.createElement("div");
        colorLabel.textContent = "Color";
        const colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.value = this.brush.color;
        colorInput.addEventListener("input", () => {
            this.brush.color = colorInput.value;
        });
        brushColorRow.appendChild(colorLabel);
        brushColorRow.appendChild(colorInput);
        brushSection.appendChild(brushSize.row);
        brushSection.appendChild(brushHardness.row);
        brushSection.appendChild(brushShapeRow);
        brushSection.appendChild(brushColorRow);
        toolbar.appendChild(brushSection);

        const center = document.createElement("div");
        center.className = "cpce-center";
        const canvasBox = document.createElement("div");
        canvasBox.className = "cpce-canvas-box";
        const previewCanvas = document.createElement("canvas");
        previewCanvas.className = "cpce-canvas";
        const overlayCanvas = document.createElement("canvas");
        overlayCanvas.className = "cpce-overlay";
        overlayCanvas.tabIndex = 0;
        const emptyMsg = document.createElement("div");
        emptyMsg.className = "cpce-empty";
        emptyMsg.textContent = "Run this node once to load incoming images.";
        canvasBox.appendChild(previewCanvas);
        canvasBox.appendChild(overlayCanvas);
        center.appendChild(canvasBox);
        center.appendChild(emptyMsg);

        const layersPanel = document.createElement("div");
        layersPanel.className = "cpce-panel cpce-layers";

        const layerSection = document.createElement("div");
        layerSection.className = "cpce-section";
        layerSection.innerHTML = `<div class="cpce-title">Layers</div>`;
        const layerList = document.createElement("div");
        layerList.className = "cpce-layer-list";
        layerSection.appendChild(layerList);
        layersPanel.appendChild(layerSection);

        const transformSection = document.createElement("div");
        transformSection.className = "cpce-section";
        transformSection.innerHTML = `<div class="cpce-title">Transform (Selected)</div>`;
        const tX = this.createRangeRow("Offset X", -1024, 1024, 0.1, 0, (v) => this.setSelectedNumeric("transform", "x", v), true);
        const tY = this.createRangeRow("Offset Y", -1024, 1024, 0.1, 0, (v) => this.setSelectedNumeric("transform", "y", v), true);
        const tPosX = this.createRangeRow("Position X", -4096, 4096, 0.1, 0, (v) => this.setSelectedNumeric("transform", "position_x", v), true);
        const tPosY = this.createRangeRow("Position Y", -4096, 4096, 0.1, 0, (v) => this.setSelectedNumeric("transform", "position_y", v), true);
        const tScaleX = this.createRangeRow("Scale X", 0.05, 4, 0.01, 1, (v) => this.setSelectedNumeric("transform", "scale_x", v), true);
        const tScaleY = this.createRangeRow("Scale Y", 0.05, 4, 0.01, 1, (v) => this.setSelectedNumeric("transform", "scale_y", v), true);
        const tRot = this.createRangeRow("Rotation", -180, 180, 1, 0, (v) => this.setSelectedNumeric("transform", "rotation", v), true);
        transformSection.appendChild(tX.row);
        transformSection.appendChild(tY.row);
        transformSection.appendChild(tPosX.row);
        transformSection.appendChild(tPosY.row);
        transformSection.appendChild(tScaleX.row);
        transformSection.appendChild(tScaleY.row);
        transformSection.appendChild(tRot.row);
        const flipRow = document.createElement("div");
        flipRow.className = "cpce-action-row";
        const flipXBtn = document.createElement("button");
        flipXBtn.className = "cpce-btn";
        flipXBtn.textContent = "Flip H";
        flipXBtn.addEventListener("click", () => this.toggleSelectedFlip("flip_x"));
        const flipYBtn = document.createElement("button");
        flipYBtn.className = "cpce-btn";
        flipYBtn.textContent = "Flip V";
        flipYBtn.addEventListener("click", () => this.toggleSelectedFlip("flip_y"));
        flipRow.appendChild(flipXBtn);
        flipRow.appendChild(flipYBtn);
        transformSection.appendChild(flipRow);
        layersPanel.appendChild(transformSection);

        const levelsSection = document.createElement("div");
        levelsSection.className = "cpce-section";
        levelsSection.innerHTML = `<div class="cpce-title">Levels (Selected)</div>`;
        const lvInBlack = this.createRangeRow("In Black", 0, 1, 0.01, 0, (v) => this.setSelectedNumeric("levels", "in_black", v), true);
        const lvGamma = this.createRangeRow("In Gamma", 0.1, 5, 0.01, 1, (v) => this.setSelectedNumeric("levels", "in_gamma", v), true);
        const lvInWhite = this.createRangeRow("In White", 0, 1, 0.01, 1, (v) => this.setSelectedNumeric("levels", "in_white", v), true);
        const lvOutBlack = this.createRangeRow("Out Black", 0, 1, 0.01, 0, (v) => this.setSelectedNumeric("levels", "out_black", v), true);
        const lvOutWhite = this.createRangeRow("Out White", 0, 1, 0.01, 1, (v) => this.setSelectedNumeric("levels", "out_white", v), true);
        levelsSection.appendChild(lvInBlack.row);
        levelsSection.appendChild(lvGamma.row);
        levelsSection.appendChild(lvInWhite.row);
        levelsSection.appendChild(lvOutBlack.row);
        levelsSection.appendChild(lvOutWhite.row);
        layersPanel.appendChild(levelsSection);

        const hueSection = document.createElement("div");
        hueSection.className = "cpce-section";
        hueSection.innerHTML = `<div class="cpce-title">Hue (Selected)</div>`;
        const hueShift = this.createRangeRow("Hue", -180, 180, 1, 0, (v) => this.setSelectedNumeric("hue", "hue", v), true);
        const hueSat = this.createRangeRow("Saturation", 0, 3, 0.01, 1, (v) => this.setSelectedNumeric("hue", "saturation", v), true);
        const hueLight = this.createRangeRow("Lightness", 0, 3, 0.01, 1, (v) => this.setSelectedNumeric("hue", "lightness", v), true);
        hueSection.appendChild(hueShift.row);
        hueSection.appendChild(hueSat.row);
        hueSection.appendChild(hueLight.row);
        layersPanel.appendChild(hueSection);

        layout.appendChild(toolbar);
        layout.appendChild(center);
        layout.appendChild(layersPanel);
        this.root.appendChild(layout);

        this.ui.layout = layout;
        this.ui.toolbar = toolbar;
        this.ui.layerList = layerList;
        this.ui.center = center;
        this.ui.previewCanvas = previewCanvas;
        this.ui.overlayCanvas = overlayCanvas;
        this.ui.emptyMsg = emptyMsg;
        this.ui.canvasBox = canvasBox;

        this.controls.transformX = tX;
        this.controls.transformY = tY;
        this.controls.transformPosX = tPosX;
        this.controls.transformPosY = tPosY;
        this.controls.transformScaleX = tScaleX;
        this.controls.transformScaleY = tScaleY;
        this.controls.transformRot = tRot;
        this.controls.levelsInBlack = lvInBlack;
        this.controls.levelsGamma = lvGamma;
        this.controls.levelsInWhite = lvInWhite;
        this.controls.levelsOutBlack = lvOutBlack;
        this.controls.levelsOutWhite = lvOutWhite;
        this.controls.hueShift = hueShift;
        this.controls.hueSat = hueSat;
        this.controls.hueLight = hueLight;
    }

    createRangeRow(label, min, max, step, initialValue, onInput, commitOnChange = false) {
        const row = document.createElement("div");
        row.className = "cpce-row cpce-row-range";
        const left = document.createElement("div");
        left.textContent = label;
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        const numberInput = document.createElement("input");
        numberInput.type = "number";
        numberInput.min = String(min);
        numberInput.max = String(max);
        numberInput.step = String(step);

        const normalize = (raw) => {
            const num = Number(raw);
            if (!Number.isFinite(num)) {
                return clamp(Number(initialValue), min, max);
            }
            return clamp(num, min, max);
        };
        const decimals = (() => {
            const asText = String(step);
            const idx = asText.indexOf(".");
            return idx === -1 ? 0 : Math.max(0, asText.length - idx - 1);
        })();
        const format = (v) => {
            if (decimals <= 0) {
                return String(Math.round(v));
            }
            return Number(v).toFixed(decimals);
        };

        const setBoth = (v) => {
            const value = normalize(v);
            input.value = format(value);
            numberInput.value = format(value);
            return value;
        };
        setBoth(initialValue);

        input.addEventListener("input", () => {
            const value = setBoth(input.value);
            onInput(value);
            if (!this.syncingControls) {
                this.render();
            }
        });
        numberInput.addEventListener("input", () => {
            const value = setBoth(numberInput.value);
            onInput(value);
            if (!this.syncingControls) {
                this.render();
            }
        });
        if (commitOnChange) {
            input.addEventListener("change", () => {
                if (this.syncingControls) {
                    return;
                }
                this.commitStateChange();
            });
            numberInput.addEventListener("change", () => {
                if (this.syncingControls) {
                    return;
                }
                const value = setBoth(numberInput.value);
                onInput(value);
                this.render();
                this.commitStateChange();
            });
        }
        row.appendChild(left);
        row.appendChild(input);
        row.appendChild(numberInput);
        return { row, input, numberInput, format };
    }

    setRangeControlValue(control, value) {
        if (!control) {
            return;
        }
        const text = control.format ? control.format(value) : String(value);
        control.input.value = text;
        if (control.numberInput) {
            control.numberInput.value = text;
        }
    }

    bindEvents() {
        const overlay = this.ui.overlayCanvas;
        overlay.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
        overlay.addEventListener("pointermove", (event) => this.handlePointerMove(event));
        overlay.addEventListener("pointerup", (event) => this.handlePointerUp(event));
        overlay.addEventListener("pointerleave", (event) => this.handlePointerUp(event));
        overlay.addEventListener("dblclick", (event) => this.handleDoubleClick(event));
        overlay.addEventListener("contextmenu", (event) => event.preventDefault());

        this.resizeObserver = new ResizeObserver(() => {
            this.fitCanvasToView();
        });
        this.resizeObserver.observe(this.ui.center);
    }

    bindApiEvents() {
        const nodeId = String(this.node.id);

        const onExecuted = (event) => {
            const detail = event?.detail || {};
            if (String(detail.node ?? "") !== nodeId) {
                return;
            }
            this.loadFromServer();
        };

        const onExecutionCached = (event) => {
            const detail = event?.detail || {};
            const nodes = Array.isArray(detail.nodes) ? detail.nodes : [];
            if (!nodes.some((id) => String(id) === nodeId)) {
                return;
            }
            this.loadFromServer();
        };

        const onExecutionComplete = () => {
            // Fallback refresh when execution events do not include per-node payload.
            this.loadFromServer();
        };

        api.addEventListener("executed", onExecuted);
        api.addEventListener("execution_cached", onExecutionCached);
        api.addEventListener("execution_success", onExecutionComplete);

        this.apiHandlers = { onExecuted, onExecutionCached, onExecutionComplete };
    }

    unbindApiEvents() {
        if (!this.apiHandlers) {
            return;
        }
        api.removeEventListener("executed", this.apiHandlers.onExecuted);
        api.removeEventListener("execution_cached", this.apiHandlers.onExecutionCached);
        api.removeEventListener("execution_success", this.apiHandlers.onExecutionComplete);
        this.apiHandlers = null;
    }

    async loadFromServer(options = {}) {
        const preferSavedState = options.preferSavedState !== false;
        const persistReloadedState = options.persistReloadedState === true;
        this.loadingFromServer = true;
        try {
            const response = await api.fetchApi(`/cp_canvas_editor/state/${encodeURIComponent(String(this.node.id))}`);
            if (!response || !response.ok) {
                this.loadingFromServer = false;
                return;
            }

            const payload = await response.json();
            if (!payload || !payload.ok) {
                this.loadingFromServer = false;
                return;
            }

            const runtime = payload.runtime;
            const savedState = payload.state;
            if (!runtime || !runtime.layers || !runtime.layers.length) {
                this.layers = [];
                this.selection = null;
                this.selectionTransform = null;
                this.history = [];
                this.historyIndex = -1;
                this.ui.emptyMsg.style.display = "block";
                this.renderLayerList();
                this.render();
                this.loadingFromServer = false;
                return;
            }

            const runtimeW = Math.max(1, parseInt(runtime.width, 10) || 1);
            const runtimeH = Math.max(1, parseInt(runtime.height, 10) || 1);
            this.canvasWidth = runtimeW;
            this.canvasHeight = runtimeH;

            this.ui.previewCanvas.width = runtimeW;
            this.ui.previewCanvas.height = runtimeH;
            this.ui.overlayCanvas.width = runtimeW;
            this.ui.overlayCanvas.height = runtimeH;

            const savedUsable = preferSavedState
                && savedState
                && savedState.canvas
                && Number(savedState.canvas.width) === runtimeW
                && Number(savedState.canvas.height) === runtimeH
                && Array.isArray(savedState.layers);

            const runtimeById = new Map();
            for (const layer of runtime.layers) {
                if (layer && layer.id) {
                    runtimeById.set(String(layer.id), layer);
                }
            }

            const incomingLayers = [];
            for (let i = 0; i < runtime.layers.length; i++) {
                const rtLayer = runtime.layers[i];
                if (!rtLayer || !rtLayer.id) {
                    continue;
                }
                const layerId = String(rtLayer.id);
                let stateLayer = null;
                if (savedUsable) {
                    stateLayer = savedState.layers.find((l) => l && String(l.id) === layerId) || savedState.layers[i] || null;
                }
                incomingLayers.push({
                    id: layerId,
                    name: stateLayer?.name || rtLayer.name || `Image ${i + 1}`,
                    visible: stateLayer?.visible !== false,
                    opacity: clamp(Number(stateLayer?.opacity ?? 1), 0, 1),
                    blend_mode: BLEND_MODES.includes(stateLayer?.blend_mode) ? stateLayer.blend_mode : "Normal",
                    transform: deepClone(stateLayer?.transform || {
                        x: 0,
                        y: 0,
                        scale_x: 1,
                        scale_y: 1,
                        rotation: 0,
                        flip_x: false,
                        flip_y: false,
                    }),
                    levels: deepClone(stateLayer?.levels || {
                        in_black: 0,
                        in_gamma: 1,
                        in_white: 1,
                        out_black: 0,
                        out_white: 1,
                    }),
                    hue: deepClone(stateLayer?.hue || {
                        hue: 0,
                        saturation: 1,
                        lightness: 1,
                    }),
                    bitmap: stateLayer?.bitmap || rtLayer.bitmap,
                });
            }

            const layerObjects = [];
            for (const src of incomingLayers) {
                const layer = this.createLayerObject(src.id, src.name, runtimeW, runtimeH);
                layer.visible = src.visible;
                layer.opacity = src.opacity;
                layer.blend_mode = src.blend_mode;
                layer.transform = {
                    x: Number(src.transform?.x ?? 0),
                    y: Number(src.transform?.y ?? 0),
                    scale_x: clamp(Number(src.transform?.scale_x ?? 1), 0.05, 4),
                    scale_y: clamp(Number(src.transform?.scale_y ?? 1), 0.05, 4),
                    rotation: clamp(Number(src.transform?.rotation ?? 0), -180, 180),
                    flip_x: !!src.transform?.flip_x,
                    flip_y: !!src.transform?.flip_y,
                };
                layer.levels = {
                    in_black: clamp(Number(src.levels?.in_black ?? 0), 0, 1),
                    in_gamma: clamp(Number(src.levels?.in_gamma ?? 1), 0.1, 5),
                    in_white: clamp(Number(src.levels?.in_white ?? 1), 0, 1),
                    out_black: clamp(Number(src.levels?.out_black ?? 0), 0, 1),
                    out_white: clamp(Number(src.levels?.out_white ?? 1), 0, 1),
                };
                if (layer.levels.in_white <= layer.levels.in_black) {
                    layer.levels.in_white = clamp(layer.levels.in_black + 0.01, 0, 1);
                }
                if (layer.levels.out_white < layer.levels.out_black) {
                    layer.levels.out_white = layer.levels.out_black;
                }
                layer.hue = {
                    hue: clamp(Number(src.hue?.hue ?? 0), -180, 180),
                    saturation: clamp(Number(src.hue?.saturation ?? 1), 0, 3),
                    lightness: clamp(Number(src.hue?.lightness ?? 1), 0, 3),
                };

                if (src.bitmap) {
                    try {
                        const img = await loadImage(src.bitmap);
                        layer.ctx.clearRect(0, 0, runtimeW, runtimeH);
                        layer.ctx.drawImage(img, 0, 0, runtimeW, runtimeH);
                    } catch (_error) {
                        const fallback = runtimeById.get(src.id);
                        if (fallback?.bitmap) {
                            try {
                                const fimg = await loadImage(fallback.bitmap);
                                layer.ctx.drawImage(fimg, 0, 0, runtimeW, runtimeH);
                            } catch (_ignored) {
                                // no-op
                            }
                        }
                    }
                }
                layerObjects.push(layer);
            }

            this.layers = layerObjects;
            this.selectedLayerId = savedUsable && savedState.selected_layer_id
                ? String(savedState.selected_layer_id)
                : (this.layers[0]?.id || null);
            if (!this.layers.find((l) => l.id === this.selectedLayerId)) {
                this.selectedLayerId = this.layers[0]?.id || null;
            }
            if (savedUsable && Object.values(TOOL).includes(savedState.tool)) {
                this.tool = savedState.tool === TOOL.TRANSFORM ? TOOL.MOVE : savedState.tool;
            }
            this.selection = null;
            this.selectionTransform = null;
            this.ui.emptyMsg.style.display = this.layers.length ? "none" : "block";

            this.resetHistory();
            this.stateVersion += 1;
            this.lastSavedVersion = this.stateVersion;

            this.renderLayerList();
            this.syncSelectedControls();
            this.setTool(this.tool);
            this.fitCanvasToView();
            this.render();

            if (persistReloadedState) {
                this.lastSavedVersion = -1;
                this.queueSave();
            }
        } catch (_error) {
            // Keep UI responsive; the node still works with default fallback.
        } finally {
            this.loadingFromServer = false;
        }
    }

    async confirmReloadInputs() {
        const ok = window.confirm("Reload images from latest Run and discard current editor state?");
        if (!ok) {
            return;
        }
        this.selection = null;
        this.selectionTransform = null;
        await this.loadFromServer({ preferSavedState: false, persistReloadedState: true });
    }

    createLayerObject(id, name, width, height) {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        return {
            id,
            name,
            visible: true,
            opacity: 1,
            blend_mode: "Normal",
            transform: {
                x: 0,
                y: 0,
                scale_x: 1,
                scale_y: 1,
                rotation: 0,
                flip_x: false,
                flip_y: false,
            },
            levels: {
                in_black: 0,
                in_gamma: 1,
                in_white: 1,
                out_black: 0,
                out_white: 1,
            },
            hue: {
                hue: 0,
                saturation: 1,
                lightness: 1,
            },
            canvas,
            ctx,
            imageRevision: 1,
            displayCacheKey: "",
            displayCanvas: null,
        };
    }

    getSelectedLayer() {
        if (!this.selectedLayerId) {
            return null;
        }
        return this.layers.find((layer) => layer.id === this.selectedLayerId) || null;
    }

    getLayerById(layerId) {
        if (!layerId) {
            return null;
        }
        return this.layers.find((layer) => layer.id === layerId) || null;
    }

    setTool(toolId) {
        if (toolId === TOOL.TRANSFORM) {
            toolId = TOOL.MOVE;
        }
        if (!Object.values(TOOL).includes(toolId)) {
            return;
        }
        this.tool = toolId;
        for (const key of Object.keys(TOOL)) {
            const id = TOOL[key];
            const btn = this.ui[`tool_${id}`];
            if (btn) {
                btn.classList.toggle("active", id === toolId);
            }
        }
        this.render();
    }

    setSelectedNumeric(groupName, keyName, value) {
        const layer = this.getSelectedLayer();
        if (!layer) {
            return;
        }
        if (groupName === "transform") {
            if (this.hasUsableSelection()) {
                const st = this.ensureSelectionTransform(layer);
                if (!st) {
                    return;
                }
                if (keyName === "x" || keyName === "y") {
                    st.transform[keyName] = Number(value);
                } else if (keyName === "position_x" || keyName === "position_y") {
                    const bounds = this.getSelectionBounds(this.selection, false);
                    if (!bounds) {
                        return;
                    }
                    const target = Number(value);
                    if (keyName === "position_x") {
                        const delta = target - bounds.x;
                        const deltaLocal = this.canvasDeltaToLayerDelta(layer, delta, 0);
                        st.transform.x += deltaLocal.x;
                        st.transform.y += deltaLocal.y;
                    } else {
                        const delta = target - bounds.y;
                        const deltaLocal = this.canvasDeltaToLayerDelta(layer, 0, delta);
                        st.transform.x += deltaLocal.x;
                        st.transform.y += deltaLocal.y;
                    }
                } else if (keyName === "rotation") {
                    st.transform.rotation = clamp(Number(value), -180, 180);
                } else {
                    st.transform[keyName] = clamp(Number(value), 0.05, 4);
                }
                this.applySelectionTransformPreview(layer, st);
                this.syncSelectedControls();
            } else if (keyName === "x" || keyName === "y") {
                layer.transform[keyName] = Number(value);
                this.syncSelectedControls();
            } else if (keyName === "position_x" || keyName === "position_y") {
                const bounds = this.getLayerCanvasBounds(layer);
                if (!bounds) {
                    return;
                }
                const target = Number(value);
                if (keyName === "position_x") {
                    layer.transform.x += target - bounds.x;
                } else {
                    layer.transform.y += target - bounds.y;
                }
                this.syncSelectedControls();
            } else if (keyName === "rotation") {
                layer.transform.rotation = clamp(Number(value), -180, 180);
                this.syncSelectedControls();
            } else {
                layer.transform[keyName] = clamp(Number(value), 0.05, 4);
                this.syncSelectedControls();
            }
        } else if (groupName === "levels") {
            if (this.hasUsableSelection()) {
                const st = this.ensureSelectionTransform(layer);
                if (!st) {
                    return;
                }
                st.levels[keyName] = clamp(Number(value), keyName === "in_gamma" ? 0.1 : 0, keyName === "in_gamma" ? 5 : 1);
                if (st.levels.in_white <= st.levels.in_black) {
                    st.levels.in_white = clamp(st.levels.in_black + 0.01, 0, 1);
                }
                if (st.levels.out_white < st.levels.out_black) {
                    st.levels.out_white = st.levels.out_black;
                }
                this.applySelectionTransformPreview(layer, st);
                this.syncSelectedControls();
                return;
            }
            layer.levels[keyName] = clamp(Number(value), keyName === "in_gamma" ? 0.1 : 0, keyName === "in_gamma" ? 5 : 1);
            if (layer.levels.in_white <= layer.levels.in_black) {
                layer.levels.in_white = clamp(layer.levels.in_black + 0.01, 0, 1);
            }
            if (layer.levels.out_white < layer.levels.out_black) {
                layer.levels.out_white = layer.levels.out_black;
            }
            this.invalidateLayerDisplay(layer, false);
            this.syncSelectedControls();
        } else if (groupName === "hue") {
            if (this.hasUsableSelection()) {
                const st = this.ensureSelectionTransform(layer);
                if (!st) {
                    return;
                }
                if (keyName === "hue") {
                    st.hue.hue = clamp(Number(value), -180, 180);
                } else {
                    st.hue[keyName] = clamp(Number(value), 0, 3);
                }
                this.applySelectionTransformPreview(layer, st);
                this.syncSelectedControls();
                return;
            }
            if (keyName === "hue") {
                layer.hue.hue = clamp(Number(value), -180, 180);
            } else {
                layer.hue[keyName] = clamp(Number(value), 0, 3);
            }
            this.invalidateLayerDisplay(layer, false);
        }
    }

    toggleSelectedFlip(key) {
        const layer = this.getSelectedLayer();
        if (!layer) {
            return;
        }
        if (this.hasUsableSelection()) {
            const st = this.ensureSelectionTransform(layer);
            if (!st) {
                return;
            }
            st.transform[key] = !st.transform[key];
            this.applySelectionTransformPreview(layer, st);
            this.syncSelectedControls();
            this.render();
            this.commitStateChange();
            return;
        }
        layer.transform[key] = !layer.transform[key];
        this.render();
        this.commitStateChange();
    }

    renderLayerList() {
        const list = this.ui.layerList;
        list.innerHTML = "";
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const item = document.createElement("div");
            item.className = "cpce-layer-item";
            if (layer.id === this.selectedLayerId) {
                item.classList.add("selected");
            }

            const head = document.createElement("div");
            head.className = "cpce-layer-head";
            const selected = document.createElement("input");
            selected.type = "radio";
            selected.name = `cpce_select_${this.node.id}`;
            selected.checked = layer.id === this.selectedLayerId;
            selected.addEventListener("change", () => {
                this.selectedLayerId = layer.id;
                this.selectionTransform = null;
                this.renderLayerList();
                this.syncSelectedControls();
                this.render();
            });
            const name = document.createElement("div");
            name.className = "cpce-layer-name";
            name.textContent = layer.name;
            name.addEventListener("click", () => {
                this.selectedLayerId = layer.id;
                this.selectionTransform = null;
                this.renderLayerList();
                this.syncSelectedControls();
                this.render();
            });
            const visible = document.createElement("input");
            visible.type = "checkbox";
            visible.checked = layer.visible;
            visible.title = "Show/Hide layer";
            visible.addEventListener("change", () => {
                layer.visible = visible.checked;
                this.render();
                this.commitStateChange();
            });
            head.appendChild(selected);
            head.appendChild(name);
            head.appendChild(visible);
            item.appendChild(head);

            const opacityRow = document.createElement("div");
            opacityRow.className = "cpce-layer-mini";
            const opacityLabel = document.createElement("div");
            opacityLabel.className = "cpce-tiny";
            opacityLabel.textContent = "Opacity";
            const opacityInput = document.createElement("input");
            opacityInput.type = "range";
            opacityInput.min = "0";
            opacityInput.max = "1";
            opacityInput.step = "0.01";
            opacityInput.value = String(layer.opacity);
            const opacityNumber = document.createElement("input");
            opacityNumber.type = "number";
            opacityNumber.min = "0";
            opacityNumber.max = "1";
            opacityNumber.step = "0.01";
            opacityNumber.value = String(layer.opacity.toFixed(2));
            const setOpacity = (rawValue, commit = false) => {
                const parsed = parseFloat(String(rawValue ?? ""));
                layer.opacity = clamp(Number.isFinite(parsed) ? parsed : layer.opacity, 0, 1);
                const text = layer.opacity.toFixed(2);
                opacityInput.value = text;
                opacityNumber.value = text;
                this.render();
                if (commit) {
                    this.commitStateChange();
                }
            };
            opacityInput.addEventListener("input", () => {
                setOpacity(opacityInput.value, false);
            });
            opacityInput.addEventListener("change", () => setOpacity(opacityInput.value, true));
            opacityNumber.addEventListener("input", () => setOpacity(opacityNumber.value, false));
            opacityNumber.addEventListener("change", () => setOpacity(opacityNumber.value, true));
            opacityRow.appendChild(opacityLabel);
            opacityRow.appendChild(opacityInput);
            opacityRow.appendChild(opacityNumber);
            item.appendChild(opacityRow);

            const blendRow = document.createElement("div");
            blendRow.className = "cpce-layer-mini";
            blendRow.style.gridTemplateColumns = "52px 1fr";
            const blendLabel = document.createElement("div");
            blendLabel.className = "cpce-tiny";
            blendLabel.textContent = "Blend";
            const blendSelect = document.createElement("select");
            for (const mode of BLEND_MODES) {
                const option = document.createElement("option");
                option.value = mode;
                option.textContent = mode;
                blendSelect.appendChild(option);
            }
            blendSelect.value = layer.blend_mode;
            blendSelect.addEventListener("change", () => {
                layer.blend_mode = BLEND_MODES.includes(blendSelect.value) ? blendSelect.value : "Normal";
                this.render();
                this.commitStateChange();
            });
            blendRow.appendChild(blendLabel);
            blendRow.appendChild(blendSelect);
            item.appendChild(blendRow);

            list.appendChild(item);
        }
    }

    syncSelectedControls() {
        const layer = this.getSelectedLayer();
        this.syncingControls = true;
        if (layer) {
            const tr = this.hasUsableSelection()
                ? (this.selectionTransform?.layerId === layer.id ? this.selectionTransform.transform : {
                    x: 0,
                    y: 0,
                    scale_x: 1,
                    scale_y: 1,
                    rotation: 0,
                })
                : layer.transform;

            const lv = this.hasUsableSelection() && this.selectionTransform?.layerId === layer.id
                ? this.selectionTransform.levels
                : layer.levels;
            const hu = this.hasUsableSelection() && this.selectionTransform?.layerId === layer.id
                ? this.selectionTransform.hue
                : layer.hue;
            const posBounds = this.hasUsableSelection()
                ? this.getSelectionBounds(this.selection, false)
                : this.getLayerCanvasBounds(layer);

            this.setRangeControlValue(this.controls.transformX, tr.x);
            this.setRangeControlValue(this.controls.transformY, tr.y);
            this.setRangeControlValue(this.controls.transformPosX, posBounds?.x ?? 0);
            this.setRangeControlValue(this.controls.transformPosY, posBounds?.y ?? 0);
            this.setRangeControlValue(this.controls.transformScaleX, tr.scale_x);
            this.setRangeControlValue(this.controls.transformScaleY, tr.scale_y);
            this.setRangeControlValue(this.controls.transformRot, tr.rotation);

            this.setRangeControlValue(this.controls.levelsInBlack, lv.in_black);
            this.setRangeControlValue(this.controls.levelsGamma, lv.in_gamma);
            this.setRangeControlValue(this.controls.levelsInWhite, lv.in_white);
            this.setRangeControlValue(this.controls.levelsOutBlack, lv.out_black);
            this.setRangeControlValue(this.controls.levelsOutWhite, lv.out_white);

            this.setRangeControlValue(this.controls.hueShift, hu.hue);
            this.setRangeControlValue(this.controls.hueSat, hu.saturation);
            this.setRangeControlValue(this.controls.hueLight, hu.lightness);
        }
        this.syncingControls = false;
    }

    resetHistory() {
        this.history = [];
        this.historyIndex = -1;
        this.recordHistorySnapshot();
        this.updateUndoRedoButtons();
    }

    createHistorySnapshot() {
        return {
            selectedLayerId: this.selectedLayerId,
            tool: this.tool,
            selection: deepClone(this.selection),
            layers: this.layers.map((layer) => {
                const data = layer.ctx.getImageData(0, 0, this.canvasWidth, this.canvasHeight);
                return {
                    id: layer.id,
                    name: layer.name,
                    visible: layer.visible,
                    opacity: layer.opacity,
                    blend_mode: layer.blend_mode,
                    transform: deepClone(layer.transform),
                    levels: deepClone(layer.levels),
                    hue: deepClone(layer.hue),
                    imageData: new ImageData(new Uint8ClampedArray(data.data), data.width, data.height),
                };
            }),
        };
    }

    applyHistorySnapshot(snapshot) {
        if (!snapshot || !Array.isArray(snapshot.layers)) {
            return;
        }

        this.historySuspended = true;
        try {
            const layerById = new Map();
            for (const layer of this.layers) {
                layerById.set(layer.id, layer);
            }

            for (const src of snapshot.layers) {
                const dst = layerById.get(src.id);
                if (!dst) {
                    continue;
                }
                dst.name = src.name;
                dst.visible = !!src.visible;
                dst.opacity = clamp(Number(src.opacity || 0), 0, 1);
                dst.blend_mode = BLEND_MODES.includes(src.blend_mode) ? src.blend_mode : "Normal";
                dst.transform = deepClone(src.transform);
                dst.levels = deepClone(src.levels);
                dst.hue = deepClone(src.hue);
                dst.ctx.putImageData(src.imageData, 0, 0);
                this.invalidateLayerDisplay(dst);
            }

            this.selectedLayerId = snapshot.selectedLayerId;
            if (!this.layers.find((layer) => layer.id === this.selectedLayerId)) {
                this.selectedLayerId = this.layers[0]?.id || null;
            }
            if (Object.values(TOOL).includes(snapshot.tool)) {
                this.tool = snapshot.tool === TOOL.TRANSFORM ? TOOL.MOVE : snapshot.tool;
            }
            this.selection = deepClone(snapshot.selection || null);
            this.selectionTransform = null;
            this.renderLayerList();
            this.syncSelectedControls();
            this.setTool(this.tool);
            this.render();
        } finally {
            this.historySuspended = false;
        }
    }

    recordHistorySnapshot() {
        if (this.historySuspended || !this.layers.length) {
            this.updateUndoRedoButtons();
            return;
        }
        const snapshot = this.createHistorySnapshot();
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        this.history.push(snapshot);
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
        this.historyIndex = this.history.length - 1;
        this.updateUndoRedoButtons();
    }

    undo() {
        if (this.historyIndex <= 0) {
            return;
        }
        this.historyIndex -= 1;
        this.applyHistorySnapshot(this.history[this.historyIndex]);
        this.updateUndoRedoButtons();
        this.markDirtyAndSave();
    }

    redo() {
        if (this.historyIndex >= this.history.length - 1) {
            return;
        }
        this.historyIndex += 1;
        this.applyHistorySnapshot(this.history[this.historyIndex]);
        this.updateUndoRedoButtons();
        this.markDirtyAndSave();
    }

    updateUndoRedoButtons() {
        this.ui.undoBtn.disabled = this.historyIndex <= 0;
        this.ui.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
    }

    commitStateChange() {
        this.recordHistorySnapshot();
        this.markDirtyAndSave();
    }

    markDirtyAndSave() {
        if (this.loadingFromServer) {
            return;
        }
        this.stateVersion += 1;
        this.queueSave();
    }

    queueSave() {
        if (this.loadingFromServer) {
            return;
        }
        if (this.saveTimer) {
            clearTimeout(this.saveTimer);
        }
        this.saveTimer = setTimeout(() => {
            this.saveTimer = null;
            this.saveStateToServer();
        }, 350);
    }

    serializeState() {
        return {
            version: 1,
            canvas: {
                width: this.canvasWidth,
                height: this.canvasHeight,
            },
            selected_layer_id: this.selectedLayerId,
            tool: this.tool,
            layers: this.layers.map((layer) => ({
                id: layer.id,
                name: layer.name,
                visible: layer.visible,
                opacity: layer.opacity,
                blend_mode: layer.blend_mode,
                transform: deepClone(layer.transform),
                levels: deepClone(layer.levels),
                hue: deepClone(layer.hue),
                bitmap: layer.canvas.toDataURL("image/png"),
            })),
        };
    }

    async saveStateToServer() {
        if (this.loadingFromServer) {
            return;
        }
        if (this.lastSavedVersion === this.stateVersion) {
            return;
        }
        const payload = {
            node_id: String(this.node.id),
            state: this.serializeState(),
        };
        try {
            const response = await api.fetchApi("/cp_canvas_editor/state", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });
            if (response?.ok) {
                this.lastSavedVersion = this.stateVersion;
            }
        } catch (_error) {
            // Avoid blocking editor flow on transient network errors.
        }
    }

    handlePointerDown(event) {
        if (!this.layers.length) {
            return;
        }
        const layer = this.getSelectedLayer();
        if (!layer) {
            return;
        }

        const point = this.getCanvasPoint(event);
        this.ui.overlayCanvas.setPointerCapture(event.pointerId);

        if (this.tool === TOOL.SELECT_LASSO) {
            this.selectionTransform = null;
            this.selection = {
                type: "lasso",
                points: [{ x: point.x, y: point.y }],
                current: { x: point.x, y: point.y },
                closed: false,
            };
            this.interaction = {
                type: "lasso_draw",
                lastX: point.x,
                lastY: point.y,
            };
            this.syncSelectedControls();
            this.render();
            return;
        }

        if (this.tool === TOOL.MOVE || this.tool === TOOL.TRANSFORM) {
            if (this.hasUsableSelection() && this.beginSelectionMove(layer, point)) {
                this.render();
                return;
            }
            this.selectionTransform = null;
            this.interaction = {
                type: "move",
                layerId: layer.id,
                startX: point.x,
                startY: point.y,
                baseX: layer.transform.x,
                baseY: layer.transform.y,
            };
            return;
        }

        if (this.tool === TOOL.BRUSH || this.tool === TOOL.ERASER) {
            this.selectionTransform = null;
            const layerPoint = this.canvasToLayerPoint(layer, point);
            this.interaction = {
                type: "paint",
                layerId: layer.id,
                lastX: layerPoint.x,
                lastY: layerPoint.y,
                erase: this.tool === TOOL.ERASER,
                changed: false,
            };
            this.paintStamp(layer, layerPoint.x, layerPoint.y, this.tool === TOOL.ERASER);
            this.interaction.changed = true;
            this.render();
            return;
        }

        if (this.tool === TOOL.SELECT_RECT) {
            this.selectionTransform = null;
            this.interaction = {
                type: "rect_selection",
                startX: point.x,
                startY: point.y,
            };
            this.selection = {
                type: "rect",
                x: point.x,
                y: point.y,
                w: 0,
                h: 0,
            };
            this.syncSelectedControls();
            this.render();
        }
    }

    handlePointerMove(event) {
        if (!this.layers.length) {
            return;
        }

        const point = this.getCanvasPoint(event);
        const interaction = this.interaction;
        const layer = interaction?.layerId ? this.getLayerById(interaction.layerId) : this.getSelectedLayer();

        if (interaction?.type === "lasso_draw" && this.selection && this.selection.type === "lasso" && !this.selection.closed) {
            const prev = this.selection.points[this.selection.points.length - 1];
            const dist = prev ? Math.hypot(point.x - prev.x, point.y - prev.y) : 0;
            if (dist >= 1) {
                this.selection.points.push({ x: point.x, y: point.y });
            }
            this.selection.current = { x: point.x, y: point.y };
            this.render();
            return;
        }

        if (!interaction || !layer) {
            return;
        }

        if (interaction.type === "move") {
            layer.transform.x = interaction.baseX + (point.x - interaction.startX);
            layer.transform.y = interaction.baseY + (point.y - interaction.startY);
            this.syncSelectedControls();
            this.render();
            return;
        }

        if (interaction.type === "move_selection") {
            const localPoint = this.canvasToLayerPoint(layer, point);
            const dx = localPoint.x - interaction.startLocal.x;
            const dy = localPoint.y - interaction.startLocal.y;
            interaction.st.transform.x = interaction.baseX + dx;
            interaction.st.transform.y = interaction.baseY + dy;
            interaction.changed = Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5;
            this.applySelectionTransformPreview(layer, interaction.st);
            this.syncSelectedControls();
            this.render();
            return;
        }

        if (interaction.type === "paint") {
            const layerPoint = this.canvasToLayerPoint(layer, point);
            this.paintLine(
                layer,
                interaction.lastX,
                interaction.lastY,
                layerPoint.x,
                layerPoint.y,
                interaction.erase
            );
            interaction.lastX = layerPoint.x;
            interaction.lastY = layerPoint.y;
            interaction.changed = true;
            this.render();
            return;
        }

        if (interaction.type === "rect_selection") {
            const x0 = interaction.startX;
            const y0 = interaction.startY;
            this.selection = {
                type: "rect",
                x: Math.min(x0, point.x),
                y: Math.min(y0, point.y),
                w: Math.abs(point.x - x0),
                h: Math.abs(point.y - y0),
            };
            this.render();
        }
    }

    handlePointerUp(event) {
        if (!this.layers.length) {
            return;
        }
        const interaction = this.interaction;
        const layer = interaction?.layerId ? this.getLayerById(interaction.layerId) : this.getSelectedLayer();

        if (interaction?.type === "lasso_draw") {
            if (this.selection?.type === "lasso" && this.selection.points.length >= 3) {
                this.selection.closed = true;
                this.selection.current = null;
                this.commitStateChange();
            } else {
                this.selection = null;
                this.selectionTransform = null;
            }
            this.syncSelectedControls();
            this.render();
        } else if (interaction?.type === "rect_selection") {
            if (this.selection && this.getSelectionBounds(this.selection, false)) {
                this.commitStateChange();
            } else {
                this.selection = null;
            }
            this.syncSelectedControls();
            this.render();
        } else if (interaction?.type === "move") {
            this.commitStateChange();
        } else if (interaction?.type === "move_selection" && interaction.changed) {
            if (layer) {
                this.invalidateLayerDisplay(layer);
            }
            this.commitStateChange();
        } else if (interaction?.type === "paint" && interaction.changed) {
            this.commitStateChange();
        }
        this.interaction = null;
        try {
            this.ui.overlayCanvas.releasePointerCapture(event.pointerId);
        } catch (_error) {
            // no-op
        }
    }

    handleDoubleClick(event) {
        if (this.tool !== TOOL.SELECT_LASSO || !this.selection || this.selection.type !== "lasso") {
            return;
        }
        if (this.selection.points.length < 3) {
            return;
        }
        this.selection.closed = true;
        this.selection.current = null;
        if (this.interaction?.type === "lasso_draw") {
            this.interaction = null;
        }
        this.syncSelectedControls();
        this.render();
        this.commitStateChange();
    }

    getCanvasPoint(event) {
        const rect = this.ui.overlayCanvas.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * this.canvasWidth;
        const y = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * this.canvasHeight;
        return {
            x: clamp(x, 0, this.canvasWidth),
            y: clamp(y, 0, this.canvasHeight),
        };
    }

    canvasToLayerPoint(layer, point) {
        const sx = (layer.transform.flip_x ? -1 : 1) * clamp(layer.transform.scale_x, 0.05, 4);
        const sy = (layer.transform.flip_y ? -1 : 1) * clamp(layer.transform.scale_y, 0.05, 4);
        const theta = (layer.transform.rotation * Math.PI) / 180;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const cx = this.canvasWidth / 2 + layer.transform.x;
        const cy = this.canvasHeight / 2 + layer.transform.y;

        const dx = point.x - cx;
        const dy = point.y - cy;

        const rx = dx * cosT + dy * sinT;
        const ry = -dx * sinT + dy * cosT;

        return {
            x: rx / sx + this.canvasWidth / 2,
            y: ry / sy + this.canvasHeight / 2,
        };
    }

    layerToCanvasPoint(layer, point) {
        const sx = (layer.transform.flip_x ? -1 : 1) * clamp(layer.transform.scale_x, 0.05, 4);
        const sy = (layer.transform.flip_y ? -1 : 1) * clamp(layer.transform.scale_y, 0.05, 4);
        const theta = (layer.transform.rotation * Math.PI) / 180;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const cx = this.canvasWidth / 2 + layer.transform.x;
        const cy = this.canvasHeight / 2 + layer.transform.y;

        const dx = (point.x - this.canvasWidth / 2) * sx;
        const dy = (point.y - this.canvasHeight / 2) * sy;

        return {
            x: dx * cosT - dy * sinT + cx,
            y: dx * sinT + dy * cosT + cy,
        };
    }

    canvasDeltaToLayerDelta(layer, deltaX, deltaY) {
        const sx = (layer.transform.flip_x ? -1 : 1) * clamp(layer.transform.scale_x, 0.05, 4);
        const sy = (layer.transform.flip_y ? -1 : 1) * clamp(layer.transform.scale_y, 0.05, 4);
        const theta = (layer.transform.rotation * Math.PI) / 180;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        return {
            x: (deltaX * cosT + deltaY * sinT) / sx,
            y: (-deltaX * sinT + deltaY * cosT) / sy,
        };
    }

    getLayerCanvasBounds(layer) {
        if (!layer) {
            return null;
        }
        const corners = [
            { x: 0, y: 0 },
            { x: this.canvasWidth, y: 0 },
            { x: this.canvasWidth, y: this.canvasHeight },
            { x: 0, y: this.canvasHeight },
        ].map((p) => this.layerToCanvasPoint(layer, p));

        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const p of corners) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }
        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
            return null;
        }
        return {
            x: minX,
            y: minY,
            w: maxX - minX,
            h: maxY - minY,
        };
    }

    mapSelectionToLayer(layer, selection) {
        if (!selection || !layer) {
            return null;
        }
        if (selection.type === "rect") {
            const x = selection.x;
            const y = selection.y;
            const w = selection.w;
            const h = selection.h;
            const corners = [
                { x, y },
                { x: x + w, y },
                { x: x + w, y: y + h },
                { x, y: y + h },
            ].map((p) => this.canvasToLayerPoint(layer, p));
            return { type: "lasso", points: corners, current: null, closed: true };
        }
        if (selection.type === "lasso" && selection.points?.length >= 3) {
            return {
                type: "lasso",
                points: selection.points.map((p) => this.canvasToLayerPoint(layer, p)),
                current: null,
                closed: !!selection.closed,
            };
        }
        return null;
    }

    mapSelectionToCanvas(layer, selection) {
        if (!selection || !layer) {
            return null;
        }
        if (selection.type === "lasso" && selection.points?.length >= 3) {
            return {
                type: "lasso",
                points: selection.points.map((p) => this.layerToCanvasPoint(layer, p)),
                current: null,
                closed: !!selection.closed,
            };
        }
        if (selection.type === "rect") {
            const x = selection.x;
            const y = selection.y;
            const w = selection.w;
            const h = selection.h;
            const corners = [
                { x, y },
                { x: x + w, y },
                { x: x + w, y: y + h },
                { x, y: y + h },
            ].map((p) => this.layerToCanvasPoint(layer, p));
            return { type: "lasso", points: corners, current: null, closed: true };
        }
        return null;
    }

    hasUsableSelection() {
        if (this.selectionTransform && this.selection) {
            return true;
        }
        if (!this.selection) {
            return false;
        }
        if (this.selection.type === "rect") {
            return !!this.getSelectionBounds(this.selection, false);
        }
        if (this.selection.type === "lasso") {
            return !!this.selection.closed && !!this.getSelectionBounds(this.selection, false);
        }
        return false;
    }

    getSelectionSignature(selection) {
        try {
            return JSON.stringify(selection || null);
        } catch (_error) {
            return "";
        }
    }

    ensureSelectionTransform(layer) {
        if (!layer || !this.hasUsableSelection()) {
            return null;
        }

        const existing = this.selectionTransform;
        if (existing && existing.layerId === layer.id) {
            return existing;
        }

        const sourceSelection = this.mapSelectionToLayer(layer, this.selection);
        if (!sourceSelection || !sourceSelection.points?.length) {
            return null;
        }
        const signature = this.getSelectionSignature(sourceSelection);

        const selectedCanvas = document.createElement("canvas");
        selectedCanvas.width = this.canvasWidth;
        selectedCanvas.height = this.canvasHeight;
        const selectedCtx = selectedCanvas.getContext("2d");
        selectedCtx.save();
        if (!this.applySelectionClip(selectedCtx, sourceSelection)) {
            selectedCtx.restore();
            return null;
        }
        selectedCtx.drawImage(layer.canvas, 0, 0);
        selectedCtx.restore();

        const baseCanvas = document.createElement("canvas");
        baseCanvas.width = this.canvasWidth;
        baseCanvas.height = this.canvasHeight;
        const baseCtx = baseCanvas.getContext("2d");
        baseCtx.drawImage(layer.canvas, 0, 0);
        baseCtx.save();
        if (!this.applySelectionClip(baseCtx, sourceSelection)) {
            baseCtx.restore();
            return null;
        }
        baseCtx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
        baseCtx.restore();

        const st = {
            layerId: layer.id,
            signature,
            sourceSelection,
            baseCanvas,
            selectedCanvas,
            transform: {
                x: 0,
                y: 0,
                scale_x: 1,
                scale_y: 1,
                rotation: 0,
                flip_x: false,
                flip_y: false,
            },
            levels: {
                in_black: 0,
                in_gamma: 1,
                in_white: 1,
                out_black: 0,
                out_white: 1,
            },
            hue: {
                hue: 0,
                saturation: 1,
                lightness: 1,
            },
        };
        this.selectionTransform = st;
        return st;
    }

    selectionToPoints(selection) {
        if (!selection) {
            return [];
        }
        if (selection.type === "lasso" && selection.points?.length) {
            return selection.points.map((p) => ({ x: p.x, y: p.y }));
        }
        if (selection.type === "rect") {
            const x = selection.x;
            const y = selection.y;
            const w = selection.w;
            const h = selection.h;
            return [
                { x, y },
                { x: x + w, y },
                { x: x + w, y: y + h },
                { x, y: y + h },
            ];
        }
        return [];
    }

    transformSelection(sourceSelection, transform) {
        const bounds = this.getSelectionBounds(sourceSelection);
        if (!bounds) {
            return deepClone(sourceSelection);
        }

        const points = this.selectionToPoints(sourceSelection);
        if (!points.length) {
            return deepClone(sourceSelection);
        }

        const cx = bounds.x + bounds.w / 2;
        const cy = bounds.y + bounds.h / 2;
        const sx = (transform.flip_x ? -1 : 1) * clamp(transform.scale_x, 0.05, 4);
        const sy = (transform.flip_y ? -1 : 1) * clamp(transform.scale_y, 0.05, 4);
        const theta = (clamp(transform.rotation, -180, 180) * Math.PI) / 180;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const tx = Number(transform.x) || 0;
        const ty = Number(transform.y) || 0;

        const transformed = points.map((p) => {
            const dx = p.x - cx;
            const dy = p.y - cy;
            const scaledX = dx * sx;
            const scaledY = dy * sy;
            const rx = scaledX * cosT - scaledY * sinT;
            const ry = scaledX * sinT + scaledY * cosT;
            return {
                x: rx + cx + tx,
                y: ry + cy + ty,
            };
        });

        return {
            type: "lasso",
            points: transformed,
            current: null,
            closed: true,
        };
    }

    getAdjustedSelectionCanvas(st) {
        const outCanvas = document.createElement("canvas");
        outCanvas.width = this.canvasWidth;
        outCanvas.height = this.canvasHeight;
        const outCtx = outCanvas.getContext("2d");
        outCtx.drawImage(st.selectedCanvas, 0, 0);

        const imageData = outCtx.getImageData(0, 0, this.canvasWidth, this.canvasHeight);
        const data = imageData.data;

        const inBlack = clamp(st.levels.in_black, 0, 1);
        const inGamma = clamp(st.levels.in_gamma, 0.1, 5);
        const inWhite = Math.max(clamp(st.levels.in_white, 0, 1), inBlack + 0.001);
        const outBlack = clamp(st.levels.out_black, 0, 1);
        const outWhite = Math.max(clamp(st.levels.out_white, 0, 1), outBlack);

        const hueShift = st.hue.hue / 360;
        const satMul = clamp(st.hue.saturation, 0, 3);
        const lightMul = clamp(st.hue.lightness, 0, 3);

        const levelsNeed = Math.abs(inBlack) > 1e-6
            || Math.abs(inGamma - 1) > 1e-6
            || Math.abs(inWhite - 1) > 1e-6
            || Math.abs(outBlack) > 1e-6
            || Math.abs(outWhite - 1) > 1e-6;
        const hueNeed = Math.abs(hueShift) > 1e-6 || Math.abs(satMul - 1) > 1e-6 || Math.abs(lightMul - 1) > 1e-6;

        if (!(levelsNeed || hueNeed)) {
            return st.selectedCanvas;
        }

        for (let i = 0; i < data.length; i += 4) {
            const a = data[i + 3];
            if (a === 0) {
                continue;
            }
            let r = data[i] / 255;
            let g = data[i + 1] / 255;
            let b = data[i + 2] / 255;

            if (levelsNeed) {
                r = outBlack + Math.pow(clamp((r - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
                g = outBlack + Math.pow(clamp((g - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
                b = outBlack + Math.pow(clamp((b - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
            }

            if (hueNeed) {
                let [h, s, v] = rgbToHsv(r, g, b);
                h = (h + hueShift) % 1;
                if (h < 0) {
                    h += 1;
                }
                s = clamp(s * satMul, 0, 1);
                v = clamp(v * lightMul, 0, 1);
                [r, g, b] = hsvToRgb(h, s, v);
            }

            data[i] = Math.round(clamp(r, 0, 1) * 255);
            data[i + 1] = Math.round(clamp(g, 0, 1) * 255);
            data[i + 2] = Math.round(clamp(b, 0, 1) * 255);
        }

        outCtx.putImageData(imageData, 0, 0);
        return outCanvas;
    }

    applySelectionTransformPreview(layer, st) {
        if (!layer || !st) {
            return;
        }

        const bounds = this.getSelectionBounds(st.sourceSelection);
        if (!bounds) {
            return;
        }

        const t = st.transform;
        const cx = bounds.x + bounds.w / 2;
        const cy = bounds.y + bounds.h / 2;
        const sx = (t.flip_x ? -1 : 1) * clamp(t.scale_x, 0.05, 4);
        const sy = (t.flip_y ? -1 : 1) * clamp(t.scale_y, 0.05, 4);
        const theta = (clamp(t.rotation, -180, 180) * Math.PI) / 180;
        const tx = Number(t.x) || 0;
        const ty = Number(t.y) || 0;
        const adjustedSelection = this.getAdjustedSelectionCanvas(st);

        layer.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
        layer.ctx.drawImage(st.baseCanvas, 0, 0);
        layer.ctx.save();
        layer.ctx.translate(cx + tx, cy + ty);
        layer.ctx.rotate(theta);
        layer.ctx.scale(sx, sy);
        layer.ctx.drawImage(adjustedSelection, -cx, -cy);
        layer.ctx.restore();

        const transformedLocal = this.transformSelection(st.sourceSelection, t);
        const transformedCanvas = this.mapSelectionToCanvas(layer, transformedLocal);
        if (transformedCanvas) {
            this.selection = transformedCanvas;
        }
        this.invalidateLayerDisplay(layer, false);
    }

    beginSelectionMove(layer, point) {
        if (!layer || !this.hasUsableSelection()) {
            return false;
        }

        const st = this.ensureSelectionTransform(layer);
        if (!st) {
            return false;
        }

        this.interaction = {
            type: "move_selection",
            layerId: layer.id,
            startX: point.x,
            startY: point.y,
            startLocal: this.canvasToLayerPoint(layer, point),
            changed: false,
            baseX: st.transform.x,
            baseY: st.transform.y,
            st,
        };
        return true;
    }

    getBrushStamp(erase) {
        const key = `${this.brush.size}|${this.brush.hardness}|${this.brush.shape}|${this.brush.color}|${erase ? 1 : 0}`;
        const cached = this.brushStampCache.get(key);
        if (cached) {
            return cached;
        }

        const size = Math.max(1, Math.round(this.brush.size));
        const radius = size / 2;
        const hardness = clamp(this.brush.hardness, 0.01, 1);
        const colorHex = this.brush.color || "#ffffff";
        const r = parseInt(colorHex.slice(1, 3), 16);
        const g = parseInt(colorHex.slice(3, 5), 16);
        const b = parseInt(colorHex.slice(5, 7), 16);

        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(size, size);
        const data = imageData.data;

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = (x + 0.5 - radius) / Math.max(radius, 1);
                const dy = (y + 0.5 - radius) / Math.max(radius, 1);
                let dist;
                if (this.brush.shape === "square") {
                    dist = Math.max(Math.abs(dx), Math.abs(dy));
                } else {
                    dist = Math.hypot(dx, dy);
                }
                let alpha = 0;
                if (dist <= hardness) {
                    alpha = 1;
                } else if (dist < 1) {
                    alpha = 1 - (dist - hardness) / Math.max(1 - hardness, 1e-6);
                }

                const idx = (y * size + x) * 4;
                data[idx] = erase ? 0 : r;
                data[idx + 1] = erase ? 0 : g;
                data[idx + 2] = erase ? 0 : b;
                data[idx + 3] = Math.round(alpha * 255);
            }
        }

        ctx.putImageData(imageData, 0, 0);
        this.brushStampCache.set(key, canvas);
        return canvas;
    }

    paintStamp(layer, x, y, erase) {
        const stamp = this.getBrushStamp(erase);
        const left = Math.round(x - stamp.width / 2);
        const top = Math.round(y - stamp.height / 2);
        layer.ctx.save();
        layer.ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
        layer.ctx.drawImage(stamp, left, top);
        layer.ctx.restore();
        this.invalidateLayerDisplay(layer);
    }

    paintLine(layer, x0, y0, x1, y1, erase) {
        const dist = Math.hypot(x1 - x0, y1 - y0);
        const step = Math.max(1, this.brush.size * 0.15);
        const steps = Math.max(1, Math.ceil(dist / step));
        for (let i = 1; i <= steps; i++) {
            const t = i / steps;
            const x = x0 + (x1 - x0) * t;
            const y = y0 + (y1 - y0) * t;
            this.paintStamp(layer, x, y, erase);
        }
    }

    getSelectionBounds(selection = this.selection, clampToCanvas = true) {
        if (!selection) {
            return null;
        }
        if (selection.type === "rect") {
            let x = Math.min(selection.x, selection.x + selection.w);
            let y = Math.min(selection.y, selection.y + selection.h);
            let w = Math.abs(selection.w);
            let h = Math.abs(selection.h);
            if (clampToCanvas) {
                x = clamp(x, 0, this.canvasWidth);
                y = clamp(y, 0, this.canvasHeight);
                w = clamp(w, 0, this.canvasWidth - x);
                h = clamp(h, 0, this.canvasHeight - y);
            }
            if (w < 1 || h < 1) {
                return null;
            }
            return { x, y, w, h };
        }
        if (selection.type === "lasso" && selection.closed && selection.points?.length >= 3) {
            let minX = this.canvasWidth;
            let minY = this.canvasHeight;
            let maxX = 0;
            let maxY = 0;
            for (const p of selection.points) {
                minX = Math.min(minX, p.x);
                minY = Math.min(minY, p.y);
                maxX = Math.max(maxX, p.x);
                maxY = Math.max(maxY, p.y);
            }
            let x = minX;
            let y = minY;
            let w = maxX - x;
            let h = maxY - y;
            if (clampToCanvas) {
                x = clamp(minX, 0, this.canvasWidth);
                y = clamp(minY, 0, this.canvasHeight);
                w = clamp(maxX - x, 0, this.canvasWidth - x);
                h = clamp(maxY - y, 0, this.canvasHeight - y);
            }
            if (w < 1 || h < 1) {
                return null;
            }
            return { x, y, w, h };
        }
        return null;
    }

    applySelectionClip(ctx, selection = this.selection) {
        if (!selection) {
            return false;
        }
        ctx.beginPath();
        if (selection.type === "rect") {
            ctx.rect(selection.x, selection.y, selection.w, selection.h);
            ctx.clip();
            return true;
        }
        if (selection.type === "lasso" && selection.closed && selection.points?.length >= 3) {
            ctx.moveTo(selection.points[0].x, selection.points[0].y);
            for (let i = 1; i < selection.points.length; i++) {
                ctx.lineTo(selection.points[i].x, selection.points[i].y);
            }
            ctx.closePath();
            ctx.clip();
            return true;
        }
        return false;
    }

    deleteSelection() {
        if (!this.selection) {
            return;
        }
        const layer = this.getSelectedLayer();
        if (!layer) {
            return;
        }
        this.selectionTransform = null;
        this.syncSelectedControls();
        const selectionLocal = this.mapSelectionToLayer(layer, this.selection);
        if (!selectionLocal) {
            return;
        }

        layer.ctx.save();
        const clipped = this.applySelectionClip(layer.ctx, selectionLocal);
        if (!clipped) {
            layer.ctx.restore();
            return;
        }
        layer.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
        layer.ctx.restore();

        this.invalidateLayerDisplay(layer);
        this.render();
        this.commitStateChange();
    }

    scaleSelection(scaleX, scaleY) {
        const layer = this.getSelectedLayer();
        const bounds = this.getSelectionBounds();
        if (!layer || !bounds) {
            return;
        }
        scaleX = clamp(Number(scaleX), 0.05, 10);
        scaleY = clamp(Number(scaleY), 0.05, 10);
        this.selectionTransform = null;

        const selectedOnly = document.createElement("canvas");
        selectedOnly.width = this.canvasWidth;
        selectedOnly.height = this.canvasHeight;
        const selectedCtx = selectedOnly.getContext("2d");
        selectedCtx.save();
        this.applySelectionClip(selectedCtx);
        selectedCtx.drawImage(layer.canvas, 0, 0);
        selectedCtx.restore();

        layer.ctx.save();
        this.applySelectionClip(layer.ctx);
        layer.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
        layer.ctx.restore();

        const newW = Math.max(1, Math.round(bounds.w * scaleX));
        const newH = Math.max(1, Math.round(bounds.h * scaleY));
        const dstX = bounds.x + (bounds.w - newW) / 2;
        const dstY = bounds.y + (bounds.h - newH) / 2;
        layer.ctx.drawImage(selectedOnly, bounds.x, bounds.y, bounds.w, bounds.h, dstX, dstY, newW, newH);

        if (this.selection?.type === "rect") {
            this.selection = {
                type: "rect",
                x: dstX,
                y: dstY,
                w: newW,
                h: newH,
            };
        } else if (this.selection?.type === "lasso" && this.selection.points?.length >= 3) {
            const cx = bounds.x + bounds.w / 2;
            const cy = bounds.y + bounds.h / 2;
            this.selection = {
                type: "lasso",
                points: this.selection.points.map((p) => ({
                    x: cx + (p.x - cx) * scaleX,
                    y: cy + (p.y - cy) * scaleY,
                })),
                current: null,
                closed: true,
            };
        }

        this.invalidateLayerDisplay(layer);
        this.render();
        this.commitStateChange();
    }

    invalidateLayerDisplay(layer, incrementRevision = true) {
        if (!layer) {
            return;
        }
        if (incrementRevision) {
            layer.imageRevision += 1;
        }
        layer.displayCacheKey = "";
        layer.displayCanvas = null;
    }

    getDisplayCanvas(layer) {
        const key = [
            layer.imageRevision,
            layer.levels.in_black,
            layer.levels.in_gamma,
            layer.levels.in_white,
            layer.levels.out_black,
            layer.levels.out_white,
            layer.hue.hue,
            layer.hue.saturation,
            layer.hue.lightness,
        ].join("|");
        if (layer.displayCanvas && layer.displayCacheKey === key) {
            return layer.displayCanvas;
        }

        const outCanvas = document.createElement("canvas");
        outCanvas.width = this.canvasWidth;
        outCanvas.height = this.canvasHeight;
        const outCtx = outCanvas.getContext("2d");
        outCtx.drawImage(layer.canvas, 0, 0);

        const imageData = outCtx.getImageData(0, 0, this.canvasWidth, this.canvasHeight);
        const data = imageData.data;

        const inBlack = clamp(layer.levels.in_black, 0, 1);
        const inGamma = clamp(layer.levels.in_gamma, 0.1, 5);
        const inWhite = Math.max(clamp(layer.levels.in_white, 0, 1), inBlack + 0.001);
        const outBlack = clamp(layer.levels.out_black, 0, 1);
        const outWhite = Math.max(clamp(layer.levels.out_white, 0, 1), outBlack);
        const hueShift = layer.hue.hue / 360;
        const satMul = clamp(layer.hue.saturation, 0, 3);
        const lightMul = clamp(layer.hue.lightness, 0, 3);

        const levelsNeed = Math.abs(inBlack) > 1e-6
            || Math.abs(inGamma - 1) > 1e-6
            || Math.abs(inWhite - 1) > 1e-6
            || Math.abs(outBlack) > 1e-6
            || Math.abs(outWhite - 1) > 1e-6;
        const hueNeed = Math.abs(hueShift) > 1e-6 || Math.abs(satMul - 1) > 1e-6 || Math.abs(lightMul - 1) > 1e-6;

        if (levelsNeed || hueNeed) {
            for (let i = 0; i < data.length; i += 4) {
                const a = data[i + 3];
                if (a === 0) {
                    continue;
                }
                let r = data[i] / 255;
                let g = data[i + 1] / 255;
                let b = data[i + 2] / 255;

                if (levelsNeed) {
                    r = outBlack + Math.pow(clamp((r - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
                    g = outBlack + Math.pow(clamp((g - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
                    b = outBlack + Math.pow(clamp((b - inBlack) / (inWhite - inBlack), 0, 1), 1 / inGamma) * (outWhite - outBlack);
                }

                if (hueNeed) {
                    let [h, s, v] = rgbToHsv(r, g, b);
                    h = (h + hueShift) % 1;
                    if (h < 0) {
                        h += 1;
                    }
                    s = clamp(s * satMul, 0, 1);
                    v = clamp(v * lightMul, 0, 1);
                    [r, g, b] = hsvToRgb(h, s, v);
                }

                data[i] = Math.round(clamp(r, 0, 1) * 255);
                data[i + 1] = Math.round(clamp(g, 0, 1) * 255);
                data[i + 2] = Math.round(clamp(b, 0, 1) * 255);
            }
            outCtx.putImageData(imageData, 0, 0);
        }

        layer.displayCacheKey = key;
        layer.displayCanvas = outCanvas;
        return outCanvas;
    }

    drawCheckerBackground(ctx) {
        const tile = 20;
        ctx.fillStyle = "#1d2024";
        ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);
        for (let y = 0; y < this.canvasHeight; y += tile) {
            for (let x = 0; x < this.canvasWidth; x += tile) {
                if (((x / tile) + (y / tile)) % 2 === 0) {
                    ctx.fillStyle = "#242932";
                } else {
                    ctx.fillStyle = "#1a1f27";
                }
                ctx.fillRect(x, y, tile, tile);
            }
        }
    }

    renderComposite() {
        const canvas = this.ui.previewCanvas;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
        this.drawCheckerBackground(ctx);

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            if (!layer.visible || layer.opacity <= 0) {
                continue;
            }
            const sourceCanvas = this.getDisplayCanvas(layer);
            ctx.save();
            ctx.globalAlpha = clamp(layer.opacity, 0, 1);
            ctx.globalCompositeOperation = BLEND_TO_CANVAS[layer.blend_mode] || "source-over";
            ctx.translate(this.canvasWidth / 2 + layer.transform.x, this.canvasHeight / 2 + layer.transform.y);
            ctx.rotate((layer.transform.rotation * Math.PI) / 180);
            const sx = (layer.transform.flip_x ? -1 : 1) * clamp(layer.transform.scale_x, 0.05, 4);
            const sy = (layer.transform.flip_y ? -1 : 1) * clamp(layer.transform.scale_y, 0.05, 4);
            ctx.scale(sx, sy);
            ctx.drawImage(sourceCanvas, -this.canvasWidth / 2, -this.canvasHeight / 2, this.canvasWidth, this.canvasHeight);
            ctx.restore();
        }
    }

    renderOverlay() {
        const overlay = this.ui.overlayCanvas;
        const ctx = overlay.getContext("2d");
        ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

        if (this.selection) {
            ctx.save();
            ctx.strokeStyle = "#75b8ff";
            ctx.fillStyle = "rgba(117, 184, 255, 0.12)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            if (this.selection.type === "rect") {
                ctx.rect(this.selection.x, this.selection.y, this.selection.w, this.selection.h);
            } else if (this.selection.type === "lasso" && this.selection.points?.length) {
                const points = this.selection.points;
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                if (this.selection.closed) {
                    ctx.closePath();
                } else if (this.selection.current) {
                    ctx.lineTo(this.selection.current.x, this.selection.current.y);
                }
            }
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }
    }

    render() {
        this.renderComposite();
        this.renderOverlay();
    }

    fitCanvasToView() {
        if (!this.canvasWidth || !this.canvasHeight) {
            return;
        }
        const panelRect = this.ui.center.getBoundingClientRect();
        const maxW = Math.max(120, panelRect.width - 24);
        const maxH = Math.max(120, panelRect.height - 24);
        const scale = Math.min(maxW / this.canvasWidth, maxH / this.canvasHeight, 1);
        const drawW = Math.max(1, Math.floor(this.canvasWidth * scale));
        const drawH = Math.max(1, Math.floor(this.canvasHeight * scale));

        this.ui.previewCanvas.style.width = `${drawW}px`;
        this.ui.previewCanvas.style.height = `${drawH}px`;
        this.ui.overlayCanvas.style.width = `${drawW}px`;
        this.ui.overlayCanvas.style.height = `${drawH}px`;
    }
}

app.registerExtension({
    name: EXTENSION_NAME,
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            injectStyles();

            const editorRoot = document.createElement("div");
            const editor = new CanvasLayerEditor(this, editorRoot);
            this.__cpCanvasEditor = editor;

            const widget = this.addDOMWidget("canvas_editor", "CP_CANVAS_EDITOR", editorRoot, {
                hideOnZoom: false,
                getMinHeight() {
                    return 980;
                },
                getMaxHeight() {
                    return 980;
                },
                getHeight() {
                    return 980;
                },
            });
            widget.serialize = false;

            const currentW = Math.max(1180, this.size?.[0] || 0);
            const currentH = Math.max(1060, this.size?.[1] || 0);
            this.setSize([currentW, currentH]);

            const prevRemoved = this.onRemoved;
            this.onRemoved = (...removedArgs) => {
                try {
                    this.__cpCanvasEditor?.destroy();
                } catch (_error) {
                    // no-op
                }
                return prevRemoved?.apply(this, removedArgs);
            };

            const prevExecuted = this.onExecuted;
            this.onExecuted = (...executedArgs) => {
                const executedResult = prevExecuted?.apply(this, executedArgs);
                this.__cpCanvasEditor?.loadFromServer();
                return executedResult;
            };

            return result;
        };
    },
});
