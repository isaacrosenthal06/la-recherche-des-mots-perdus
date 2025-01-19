import express from 'express';
import path from 'path';
import cors from 'cors';
import bodyParser from 'body-parser';
import booksRoute from './routes/getBooks';

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Routes
app.use('/books', booksRoute);

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
