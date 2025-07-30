import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import { chatting } from "./query-for-backend";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

app.post("/ask", async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Question is required" });

  try {
    const answer = await chatting(question);
    res.json({ answer });
  } catch (e) {
    console.error("Error during question handling:", e);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
