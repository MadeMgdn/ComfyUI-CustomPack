import { app } from "/scripts/app.js";

const TARGET_NODE_NAME = "MathExpression|pysssss";
const ALLOWED_INPUT_TYPES = "INT,FLOAT,IMAGE,LATENT";

app.registerExtension({
    name: "ComfyUI.CustomPack.MathExpression",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            for (const input of this.inputs || []) {
                if (["a", "b", "c"].includes(input.name)) {
                    input.type = ALLOWED_INPUT_TYPES;
                }
            }

            const expressionWidget = (this.widgets || []).find(
                (widget) => widget.name === "expression",
            );
            if (expressionWidget) {
                expressionWidget.dynamicPrompts = false;
            }

            return result;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = onDrawForeground?.apply(this, arguments);
            const output = app.nodeOutputs?.[String(this.id)];
            const value = output?.value?.[0];

            if (!this.flags?.collapsed && value !== undefined) {
                const text = String(value);
                ctx.save();
                ctx.font = "bold 12px sans-serif";
                ctx.fillStyle = "dodgerblue";
                const size = ctx.measureText(text);
                ctx.fillText(
                    text,
                    this.size[0] - size.width - 5,
                    LiteGraph.NODE_SLOT_HEIGHT * 3,
                );
                ctx.restore();
            }

            return result;
        };
    },
});
