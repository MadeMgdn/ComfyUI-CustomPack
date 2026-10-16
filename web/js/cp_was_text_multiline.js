import { app } from "/scripts/app.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.WASTextMultiline";
const TARGET_NODE_NAME = "Text Multiline";
const TEXT_WIDGET_NAME = "text";
const TOGGLE_WIDGET_NAME = "dynamic_prompts";

function widgetValue(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name)?.value;
}

app.registerExtension({
    name: EXTENSION_NAME,

    nodeCreated(node) {
        if ((node?.comfyClass || node?.type) !== TARGET_NODE_NAME) {
            return;
        }

        const widget = (node.widgets || []).find((candidate) => (
            candidate.name === TEXT_WIDGET_NAME
        ));
        if (!widget || widget.__cpWasDynamicPrompts) {
            return;
        }

        widget.__cpWasDynamicPrompts = true;
        const serializeWithDynamicPrompts = widget.serializeValue?.bind(widget);

        widget.serializeValue = (workflowNode, widgetIndex) => {
            if (widgetValue(node, TOGGLE_WIDGET_NAME) === false) {
                if (workflowNode?.widgets_values) {
                    workflowNode.widgets_values[widgetIndex] = widget.value;
                }
                return widget.value;
            }

            return serializeWithDynamicPrompts
                ? serializeWithDynamicPrompts(workflowNode, widgetIndex)
                : widget.value;
        };
    },
});
