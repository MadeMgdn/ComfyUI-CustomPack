import { app } from "/scripts/app.js";

const TARGET_NODE_NAME = "SimpleCalculatorKJ";
const VARIABLE_NAMES = [..."abcdefghijk"];
const VARIABLE_TYPE = "INT,FLOAT,BOOLEAN";
const MINIMUM_INPUTS = 2;

function variableInput(node, name) {
    return (node.inputs || []).find((input) => input.name === name);
}

function reconcileVariableInputs(node) {
    if (node.__cpCalculatorReconciling) {
        return;
    }

    node.__cpCalculatorReconciling = true;
    try {
        for (const name of VARIABLE_NAMES.slice(0, MINIMUM_INPUTS)) {
            if (!variableInput(node, name)) {
                node.addInput(name, VARIABLE_TYPE);
            }
        }

        let highestConnected = -1;
        for (let index = 0; index < VARIABLE_NAMES.length; index += 1) {
            const input = variableInput(node, VARIABLE_NAMES[index]);
            if (input?.link !== null && input?.link !== undefined) {
                highestConnected = index;
            }
        }

        const desiredCount = Math.min(
            VARIABLE_NAMES.length,
            Math.max(MINIMUM_INPUTS, highestConnected + 2),
        );

        for (const name of VARIABLE_NAMES.slice(0, desiredCount)) {
            let input = variableInput(node, name);
            if (!input) {
                node.addInput(name, VARIABLE_TYPE);
                input = variableInput(node, name);
            }
            if (input) {
                input.type = VARIABLE_TYPE;
                input.label = name;
            }
        }

        for (let index = VARIABLE_NAMES.length - 1; index >= desiredCount; index -= 1) {
            const inputIndex = (node.inputs || []).findIndex(
                (input) => input.name === VARIABLE_NAMES[index],
            );
            if (inputIndex >= 0 && node.inputs[inputIndex].link == null) {
                node.removeInput(inputIndex);
            }
        }
    } finally {
        node.__cpCalculatorReconciling = false;
    }
}

function scheduleReconcile(node) {
    setTimeout(() => {
        reconcileVariableInputs(node);
        node.setDirtyCanvas?.(true, true);
    }, 0);
}

app.registerExtension({
    name: "ComfyUI.CustomPack.SimpleCalculatorKJ",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            scheduleReconcile(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            scheduleReconcile(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            scheduleReconcile(this);
            return result;
        };
    },
});
