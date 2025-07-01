# 🤖 AI Tic Tac Toe using NEAT

This is a smart **Tic Tac Toe** game implemented in **Python** with **NEAT (NeuroEvolution of Augmenting Topologies)** to evolve AI agents that learn how to play optimally. It supports multiple play modes including:

- 👤 **Player vs AI**
- 🤖 **AI vs AI**
- 👥 (Optional) 2-Player local mode

The AI learns strategies over generations and improves gameplay using evolutionary neural networks.

---

## 🧠 How It Works

- Uses the **NEAT-Python** library to train neural networks to play Tic Tac Toe.
- The game board is converted into a numerical format and passed as input to the NEAT-based neural network.
- The network outputs the next best move based on training and past evolution.

---

## 🎮 Game Modes

1. **Player vs AI**  
   Train your brain against an AI trained over many generations.

2. **AI vs AI**  
   Watch NEAT-evolved agents battle each other.

3. **Training Mode**  
   Run the training loop to evolve smarter AI agents over time.

---

## 🛠 Tech Stack

- **Python 3**
- **NEAT-Python** for AI evolution
- **Pygame** for GUI and interaction
- **NumPy** for board logic and representation

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/ChinmayBansal010/AI-tictactoe
cd AI-tictactoe
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Game
```bash
python main.py
```

---

## 📂 Folder Structure

```
AI-tictactoe/
├── main.py                # Main game loop and UI
├── neat-config.txt        # NEAT config file
├── ai_player.py           # NEAT training logic
├── game.py                # Game board and win-check logic
├── human_player.py        # Human input logic
├── visualize.py           # NEAT visualization tools
├── models/                # Trained genomes (optional)
```

---

## 📄 License

This project is licensed for **educational and research use** only. Commercial use is not allowed without explicit permission.

---

## 🙋‍♂️ Author

**Chinmay Bansal**  
📧 chinmay8521@gmail.com  
🔗 GitHub: [@ChinmayBansal010](https://github.com/ChinmayBansal010)

---

## ⭐ Contributions

Feel free to fork, star, and open issues or pull requests for enhancements. Feedback is welcome!
