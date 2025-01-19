import express from 'express';
import { Book } from '../models/book';
import { queryDB } from '../utils/db';

const router = express.Router();

const qry: string = 'SELECT pk, title, author FROM books'

// GET /books
router.get('/', async (req, res) => {

  try {
    const bookList: Book[] = await queryDB(qry);
    if (bookList.length > 0)  {

      res.json({
        success: true,
        message: 'Books successfully pulled',
        data: bookList,
        errors: null,
      });
    } else {
      res.json({
        success: true,
        message: 'No books available',
        data: bookList,
      });
    }

  } catch (error) {

    console.error('Error fetching books:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch books. Try again some other time.',
      data: null,
      errors: error,
    });
  }

});

export default router;