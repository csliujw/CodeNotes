/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/draw.ts":
/*!*********************!*\
  !*** ./src/draw.ts ***!
  \*********************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.drawState = exports.drawLinkedList = exports.ctx = exports.canvas = void 0;
exports.canvas = document.getElementById("-ht--canvas");
exports.canvas.height = 400; // px
exports.canvas.width = 600;
exports.ctx = exports.canvas.getContext("2d");
exports.ctx.strokeStyle = "black";
exports.ctx.lineWidth = 2;
var nodeSize = 20;
var midSize = Math.trunc(nodeSize / 2);
var arrowSize = 12;
var arrowHeadSize = 6;
var space = 3;
var turtleRadius = 6;
var hareRadius = 8;
var drawTurtle = function (centerX, centerY) {
    exports.ctx.fillStyle = "#67cdcc";
    exports.ctx.beginPath();
    exports.ctx.arc(centerX, centerY, turtleRadius, 0, 2 * Math.PI);
    exports.ctx.fill();
};
var drawHare = function (centerX, centerY) {
    exports.ctx.fillStyle = "#f08d49";
    exports.ctx.beginPath();
    exports.ctx.arc(centerX, centerY, hareRadius, 0, 2 * Math.PI);
    exports.ctx.fill();
};
var drawRunners = function (startX, startY) {
    if (nodeCount == hareNode)
        drawHare(startX, startY);
    if (nodeCount == turtleNode)
        drawTurtle(startX, startY);
};
var drawSquare = function (x, y) {
    exports.ctx.lineWidth = 1;
    exports.ctx.strokeRect(x - midSize, y - midSize, nodeSize, nodeSize);
};
var convertToRad = function (angle) { return (angle * Math.PI) / 180; };
var drawArrow = function (startX, startY, angle) {
    var angleDeg = angle;
    angle = convertToRad(angle); // angle in degrees
    var deltaX = Math.round(arrowSize * Math.cos(angle));
    var deltaY = Math.round(arrowSize * Math.sin(angle));
    var endX = startX + deltaX;
    var endY = startY + deltaY;
    var angleHead1 = convertToRad(angleDeg + 180 - 45);
    var headX1 = endX + arrowHeadSize * Math.cos(angleHead1);
    var headY1 = endY + arrowHeadSize * Math.sin(angleHead1);
    var angleHead2 = convertToRad(angleDeg + 180 + 45);
    var headX2 = endX + arrowHeadSize * Math.cos(angleHead2);
    var headY2 = endY + arrowHeadSize * Math.sin(angleHead2);
    exports.ctx.lineWidth = 1;
    exports.ctx.beginPath();
    exports.ctx.moveTo(startX, startY);
    exports.ctx.lineTo(endX, endY);
    exports.ctx.lineTo(headX1, headY1);
    exports.ctx.moveTo(endX, endY);
    exports.ctx.lineTo(headX2, headY2);
    exports.ctx.stroke();
    return [endX, endY];
};
var drawSquareWithArrow = function (centerX, centerY, angle) {
    angle = angle % 360;
    drawSquare(centerX, centerY);
    if (angle >= -45 && angle <= 45)
        return drawArrow(centerX + midSize + space, centerY, angle);
    else if (angle >= 45 && angle <= 135)
        return drawArrow(centerX, centerY + midSize + space, angle);
    else if (angle >= 135 && angle <= 225)
        return drawArrow(centerX - midSize - space, centerY, angle);
    else
        return drawArrow(centerX, centerY - midSize - space, angle);
};
var drawSquareLine = function (startX, startY, n, finalArrow) {
    var _a, _b;
    if (finalArrow === void 0) { finalArrow = false; }
    for (var i = 0; i < n - 1; i++) {
        drawRunners(startX, startY);
        _a = drawSquareWithArrow(startX, startY, 0), startX = _a[0], startY = _a[1];
        startX += 2 * space + midSize;
        nodeCount++;
    }
    n && drawRunners(startX, startY);
    if (n && finalArrow) {
        _b = drawSquareWithArrow(startX, startY, 0), startX = _b[0], startY = _b[1];
        startX += 2 * space + midSize;
    }
    else
        drawSquare(startX, startY);
    n && nodeCount++;
    return [startX, startY];
};
var getArrowAngle = function (angle, deltaAngle) {
    return Math.round(angle + 180 - (180 - deltaAngle) / 2);
};
var getStartXY = function (pos, radius, delta) {
    var start = -180;
    var angle = start + delta * pos;
    return [
        Math.round(radius * Math.cos(convertToRad(angle))),
        Math.round(radius * Math.sin(convertToRad(angle))),
    ];
};
var drawSquareWithCircleArrow = function (centerX, centerY, _) {
    drawSquare(centerX, centerY);
    exports.ctx.beginPath();
    exports.ctx.lineWidth = 1;
    var startAngle = convertToRad(-135);
    var endAngle = convertToRad(135);
    exports.ctx.arc(centerX + nodeSize, centerY, nodeSize, startAngle, endAngle);
    var arrowEndX = centerX + Math.round((1 + Math.cos(endAngle)) * nodeSize);
    var arrowEndY = centerY + Math.round(Math.sin(endAngle) * nodeSize);
    exports.ctx.lineTo(arrowEndX + arrowHeadSize, arrowEndY);
    exports.ctx.moveTo(arrowEndX, arrowEndY);
    exports.ctx.lineTo(arrowEndX + (arrowHeadSize * 3) / 5, arrowEndY + (arrowHeadSize * 4) / 5);
    exports.ctx.stroke();
    exports.ctx.lineWidth = 2;
};
var drawCycle = function (startX, startY, n) {
    var _a;
    var radius = Math.round(n / 2) * nodeSize;
    var centerX = startX + radius;
    var centerY = startY;
    var deltaAngle = 360 / n;
    var cycleAngle = -180;
    var drawSquareFunction = n === 1 ? drawSquareWithCircleArrow : drawSquareWithArrow;
    for (var i = 0; i < n; i++) {
        var arrowAngle = getArrowAngle(cycleAngle, deltaAngle);
        _a = getStartXY(i, radius, deltaAngle), startX = _a[0], startY = _a[1];
        startX += centerX;
        startY += centerY;
        drawRunners(startX, startY);
        drawSquareFunction(startX, startY, arrowAngle);
        cycleAngle += deltaAngle;
        nodeCount++;
    }
};
var drawLinkedList = function (before, cycle) {
    exports.ctx.fillStyle = '#eee'
    exports.ctx.fillRect(0,0,600,400)
    var _a;
    var width = before * (nodeSize + arrowSize + 2 * space) + cycle * nodeSize;
    var startX = Math.round((exports.ctx.canvas.width - width) / 2);
    var startY = Math.round(exports.ctx.canvas.height / 2);
    drawArrow(startX - arrowSize - space - midSize, startY, 0);
    _a = drawSquareLine(startX, startY, before, !!cycle), startX = _a[0], startY = _a[1];
    drawCycle(startX, startY, cycle);
};
exports.drawLinkedList = drawLinkedList;
var drawState = function (size, cycle) {
    nodeCount = 0;
    // turtleNode, hareNode: global vars, previously updated
    exports.ctx.clearRect(0, 0, exports.canvas.width, exports.canvas.height);
    (0, exports.drawLinkedList)(size - cycle, cycle); // Max 25
};
exports.drawState = drawState;


/***/ }),

/***/ "./src/main.ts":
/*!*********************!*\
  !*** ./src/main.ts ***!
  \*********************/
/***/ (function(__unused_webpack_module, exports, __webpack_require__) {


var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", ({ value: true }));
var draw_1 = __webpack_require__(/*! ./draw */ "./src/draw.ts");
__webpack_require__.g.nodeCount = 0;
__webpack_require__.g.turtleNode = -1;
__webpack_require__.g.hareNode = -1;
var secondPhase = false;
var startSecondPhase = false;
var start = true;
var finished = false;
var sizeInput = document.getElementById("-ht--ll-size");
var cycleInput = document.getElementById("-ht--cycle-size");
var status = document.getElementById("-ht--status-content");
var startDrawing = function () {
    var size = Number(sizeInput.value);
    var cycle = Number(cycleInput.value);
    status.textContent = "未初始化";
    start = true;
    secondPhase = false;
    finished = false;
    turtleNode = -1;
    hareNode = -2;
    (0, draw_1.drawState)(size, cycle);
};
var setStatus = function () {
    if (!start && !finished && !secondPhase)
        status.textContent =
            "阶段1 - 检测链表是否有环";
    else if (finished && !secondPhase)
        status.textContent = "没有找到环. 结束.";
    else if (!finished && secondPhase && startSecondPhase)
        status.textContent = "阶段1 - 找到环了!";
    else if (!finished && secondPhase && !startSecondPhase)
        status.textContent = "阶段2 - 找环起点";
    else if (finished)
        status.textContent = "阶段2 - 找到环起点了! 结束.";
};
var nextStep = function () { return __awaiter(void 0, void 0, void 0, function () {
    var size, cycle, beforeCycle, getNextStep;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                size = Number(sizeInput.value);
                cycle = Number(cycleInput.value);
                beforeCycle = size - cycle;
                if (finished)
                    return [2 /*return*/];
                if (startSecondPhase) {
                    startSecondPhase = false;
                    (0, draw_1.drawState)(size, cycle);
                    setStatus();
                    return [2 /*return*/];
                }
                getNextStep = function (currNode) {
                    return currNode == size - 1 && cycle ? beforeCycle : currNode + 1;
                };
                turtleNode = getNextStep(turtleNode);
                hareNode = getNextStep(hareNode);
                (0, draw_1.drawState)(size, cycle);
                if (!secondPhase)
                    hareNode = getNextStep(hareNode);
                if (!start) return [3 /*break*/, 1];
                (0, draw_1.drawState)(size, cycle);
                return [3 /*break*/, 3];
            case 1: return [4 /*yield*/, new Promise(function (res) {
                    return setTimeout(function () {
                        if (turtleNode >= size || hareNode >= size) {
                            finished = true;
                            setStatus();
                        }
                        (0, draw_1.drawState)(size, cycle);
                        res(1);
                    }, 300);
                })];
            case 2:
                _a.sent();
                _a.label = 3;
            case 3:
                if (turtleNode == hareNode) {
                    if (start)
                        start = false;
                    else if (!secondPhase) {
                        secondPhase = true;
                        startSecondPhase = true;
                        turtleNode = 0;
                    }
                    else {
                        finished = true;
                    }
                }
                setStatus();
                return [2 /*return*/];
        }
    });
}); };
var updateSize = function (ev) {
    var span = document.getElementById("-ht--ll-size-value");
    span.textContent = ev.target.value;
    cycleInput.max = ev.target.value;
};
var updateCycle = function (ev) {
    var span = (document.getElementById("-ht--cycle-size-value"));
    var maxCycle = Number(sizeInput.value);
    var cycleSize = Number(cycleInput.value);
    span.textContent = cycleSize <= maxCycle ? cycleInput.value : sizeInput.value;
    cycleInput.value = cycleSize <= maxCycle ? cycleInput.value : sizeInput.value;
};
sizeInput.value = "10";
cycleInput.value = "0";
updateSize({ target: sizeInput });
startDrawing();
sizeInput.addEventListener("change", updateSize);
sizeInput.addEventListener("change", startDrawing);
cycleInput.addEventListener("change", updateCycle);
cycleInput.addEventListener("change", startDrawing);
document.getElementById("-ht--next-button").addEventListener("click", nextStep);
document
    .getElementById("-ht--reset-button")
    .addEventListener("click", startDrawing);


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/global */
/******/ 	(() => {
/******/ 		__webpack_require__.g = (function() {
/******/ 			if (typeof globalThis === 'object') return globalThis;
/******/ 			try {
/******/ 				return this || new Function('return this')();
/******/ 			} catch (e) {
/******/ 				if (typeof window === 'object') return window;
/******/ 			}
/******/ 		})();
/******/ 	})();
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module is referenced by other modules so it can't be inlined
/******/ 	var __webpack_exports__ = __webpack_require__("./src/main.ts");
/******/ 	
/******/ })()
;