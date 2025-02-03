"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const db_1 = require("../utils/db");
const router = express_1.default.Router();
const qry = 'SELECT pk, title, author FROM books';
// GET /books
router.get('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const bookList = yield (0, db_1.queryDB)(qry);
        if (bookList.length > 0) {
            res.json({
                success: true,
                message: 'Books successfully pulled',
                data: bookList,
                errors: null,
            });
        }
        else {
            res.json({
                success: true,
                message: 'No books available',
                data: bookList,
            });
        }
    }
    catch (error) {
        console.error('Error fetching books:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to fetch books. Try again some other time.',
            data: null,
            errors: error,
        });
    }
}));
exports.default = router;
