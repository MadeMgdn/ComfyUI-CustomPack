import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.Steudio.UI";
const TARGET_NODE_NAMES = new Set([
    "Ratio Calculator",
    "Sequence Generator",
    "Display UI",
    "CP ShowAnything",
]);

app.registerExtension({
    name: EXTENSION_NAME,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODE_NAMES.has(nodeData.name)) {
            return;
        }

        function populate(text) {
            if (this.widgets) {
                const isConvertedWidget = +!!this.inputs?.[0]?.widget;
                for (let index = isConvertedWidget; index < this.widgets.length; index++) {
                    this.widgets[index].onRemove?.();
                }
                this.widgets.length = isConvertedWidget;
            }

            const widget = ComfyWidgets.STRING(
                this,
                "text_box",
                ["STRING", { multiline: true }],
                app,
            ).widget;

            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.6;

            if (nodeData.name === "Display UI" && Array.isArray(text)) {
                widget.value = text.map((line) => `${line}\n`).join("\n");
            } else {
                widget.value = Array.isArray(text) ? text.join("") : (text ?? "");
            }
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message?.text);
        };

        const values = Symbol("steudioWidgetValues");
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[values] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const widgetValues = this[values];

            if (widgetValues?.length) {
                requestAnimationFrame(() => {
                    populate.call(this, widgetValues[0]);
                });
            }
        };
    },
});
