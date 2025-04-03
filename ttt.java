import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class ttt {

    // Neural network parameters.
    private static final int NN_INPUT_SIZE = 18;
    private static final int NN_HIDDEN_SIZE = 100;
    private static final int NN_OUTPUT_SIZE = 9;
    private static final float LEARNING_RATE = 0.1f;

    // Game board representation.
    static class GameState {
        char[] board = new char[9]; // Can be '.' (empty) or 'X', 'O'.
        int currentPlayer; // 0 for player (X), 1 for computer (O).

        GameState() {
            Arrays.fill(board, '.');
            currentPlayer = 0; // Player (X) goes first
        }
    }

    /* Neural network structure. For simplicity we have just
     * one hidden layer and fixed sizes (see defines above).
     * However for this problem going deeper than one hidden layer
     * is useless. */
    static class NeuralNetwork {
        // Weights and biases.
        float[] weightsIH = new float[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
        float[] weightsHO = new float[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
        float[] biasesH = new float[NN_HIDDEN_SIZE];
        float[] biasesO = new float[NN_OUTPUT_SIZE];

        // Activations are part of the structure itself for simplicity.
        float[] inputs = new float[NN_INPUT_SIZE];
        float[] hidden = new float[NN_HIDDEN_SIZE];
        float[] rawLogits = new float[NN_OUTPUT_SIZE]; // Outputs before softmax().
        float[] outputs = new float[NN_OUTPUT_SIZE]; // Outputs after softmax().
    }

    /* ReLU activation function */
    static float relu(float x) {
        return x > 0 ? x : 0;
    }

    /* Derivative of ReLU activation function */
    static float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    /* Initialize a neural network with random weights, we should
     * use something like He weights since we use RELU, but we don't
     * care as this is a trivial example. */
    static void initNeuralNetwork(NeuralNetwork nn, Random random) {
        // Initialize weights with random values between -0.5 and 0.5
        for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++) {
            nn.weightsIH[i] = random.nextFloat() - 0.5f;
        }

        for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++) {
            nn.weightsHO[i] = random.nextFloat() - 0.5f;
        }

        for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
            nn.biasesH[i] = random.nextFloat() - 0.5f;
        }

        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            nn.biasesO[i] = random.nextFloat() - 0.5f;
        }
    }

    /* Apply softmax activation function to an array input, and
     * set the result into output. */
    static void softmax(float[] input, float[] output) {
        /* Find maximum value then subtract it to avoid
         * numerical stability issues with exp(). */
        float maxVal = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > maxVal) {
                maxVal = input[i];
            }
        }

        // Calculate exp(x_i - max) for each element and sum.
        float sum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i] - maxVal);
            sum += output[i];
        }

        // Normalize to get probabilities.
        if (sum > 0) {
            for (int i = 0; i < input.length; i++) {
                output[i] /= sum;
            }
        } else {
            /* Fallback in case of numerical issues, just provide
             * a uniform distribution. */
            for (int i = 0; i < input.length; i++) {
                output[i] = 1.0f / input.length;
            }
        }
    }

    /* Neural network forward pass (inference). We store the activations
     * so we can also do backpropagation later. */
    static void forwardPass(NeuralNetwork nn, float[] inputs) {
        // Copy inputs.
        System.arraycopy(inputs, 0, nn.inputs, 0, NN_INPUT_SIZE);

        // Input to hidden layer.
        for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
            float sum = nn.biasesH[i];
            for (int j = 0; j < NN_INPUT_SIZE; j++) {
                sum += inputs[j] * nn.weightsIH[j * NN_HIDDEN_SIZE + i];
            }
            nn.hidden[i] = relu(sum);
        }

        // Hidden to output (raw logits).
        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            nn.rawLogits[i] = nn.biasesO[i];
            for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
                nn.rawLogits[i] += nn.hidden[j] * nn.weightsHO[j * NN_OUTPUT_SIZE + i];
            }
        }

        // Apply softmax to get the final probabilities.
        softmax(nn.rawLogits, nn.outputs);
    }

    /* Show board on screen in ASCII "art"... */
    static void displayBoard(GameState state) {
        for (int row = 0; row < 3; row++) {
            // Display the board symbols.
            System.out.printf("%c%c%c ", state.board[row * 3], state.board[row * 3 + 1],
                    state.board[row * 3 + 2]);

            // Display the position numbers for this row, for the poor human.
            System.out.printf("%d%d%d\n", row * 3, row * 3 + 1, row * 3 + 2);
        }
        System.out.println();
    }

    /* Convert board state to neural network inputs. Note that we use
     * a peculiar encoding I described here:
     * https://www.youtube.com/watch?v=EXbgUXt8fFU
     *
     * Instead of one-hot encoding, we can represent N different categories
     * as different bit patterns. In this specific case it's trivial:
     *
     * 00 = empty
     * 10 = X
     * 01 = O
     *
     * Two inputs per symbol instead of 3 in this case, but in the general case
     * this reduces the input dimensionality A LOT.
     *
     * LEARNING OPPORTUNITY: You may want to learn (if not already aware) of
     * different ways to represent non scalar inputs in neural networks:
     * One hot encoding, learned embeddings, and even if it's just my random
     * experiment this "permutation coding" that I'm using here.
     */
    static void boardToInputs(GameState state, float[] inputs) {
        for (int i = 0; i < 9; i++) {
            if (state.board[i] == '.') {
                inputs[i * 2] = 0;
                inputs[i * 2 + 1] = 0;
            } else if (state.board[i] == 'X') {
                inputs[i * 2] = 1;
                inputs[i * 2 + 1] = 0;
            } else { // 'O'
                inputs[i * 2] = 0;
                inputs[i * 2 + 1] = 1;
            }
        }
    }

    /* Check if the game is over (win or tie).
     * Very brutal but fast enough. */
    static boolean checkGameOver(GameState state, StringBuilder winner) {
        // Check rows.
        for (int i = 0; i < 3; i++) {
            if (state.board[i * 3] != '.' &&
                    state.board[i * 3] == state.board[i * 3 + 1] &&
                    state.board[i * 3 + 1] == state.board[i * 3 + 2]) {
                winner.append(state.board[i * 3]);
                return true;
            }
        }

        // Check columns.
        for (int i = 0; i < 3; i++) {
            if (state.board[i] != '.' &&
                    state.board[i] == state.board[i + 3] &&
                    state.board[i + 3] == state.board[i + 6]) {
                winner.append(state.board[i]);
                return true;
            }
        }

        // Check diagonals.
        if (state.board[0] != '.' &&
                state.board[0] == state.board[4] &&
                state.board[4] == state.board[8]) {
            winner.append(state.board[0]);
            return true;
        }
        if (state.board[2] != '.' &&
                state.board[2] == state.board[4] &&
                state.board[4] == state.board[6]) {
            winner.append(state.board[2]);
            return true;
        }

        // Check for tie (no free tiles left).
        int emptyTiles = 0;
        for (int i = 0; i < 9; i++) {
            if (state.board[i] == '.') emptyTiles++;
        }
        if (emptyTiles == 0) {
            winner.append('T'); // Tie
            return true;
        }

        return false; // Game continues.
    }

    /* Get the best move for the computer using the neural network.
     * Note that there is no complex sampling at all, we just get
     * the output with the highest value THAT has an empty tile. */
    static int getComputerMove(GameState state, NeuralNetwork nn, boolean displayProbs) {
        float[] inputs = new float[NN_INPUT_SIZE];

        boardToInputs(state, inputs);
        forwardPass(nn, inputs);

        // Find the highest probability value and best legal move.
        float highestProb = -1.0f;
        int highestProbIdx = -1;
        int bestMove = -1;
        float bestLegalProb = -1.0f;

        for (int i = 0; i < 9; i++) {
            // Track highest probability overall.
            if (nn.outputs[i] > highestProb) {
                highestProb = nn.outputs[i];
                highestProbIdx = i;
            }

            // Track best legal move.
            if (state.board[i] == '.' &&
                    (bestMove == -1 || nn.outputs[i] > bestLegalProb)) {
                bestMove = i;
                bestLegalProb = nn.outputs[i];
            }
        }

        // That's just for debugging. It's interesting to show to user
        // in the first iterations of the game, since you can see how initially
        // the net picks illegal moves as best, and so forth.
        if (displayProbs) {
            System.out.println("Neural network move probabilities:");
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    int pos = row * 3 + col;

                    // Print probability as percentage.
                    System.out.printf("%5.1f%%", nn.outputs[pos] * 100.0f);

                    // Add markers.
                    if (pos == highestProbIdx) {
                        System.out.print("*"); // Highest probability overall.
                    }
                    if (pos == bestMove) {
                        System.out.print("#"); // Selected move (highest valid probability).
                    }
                    System.out.print(" ");
                }
                System.out.println();
            }

            // Sum of probabilities should be 1.0, hopefully.
            // Just debugging.
            float totalProb = 0.0f;
            for (int i = 0; i < 9; i++)
                totalProb += nn.outputs[i];
            System.out.printf("Sum of all probabilities: %.2f\n\n", totalProb);
        }
        return bestMove;
    }

    /* Backpropagation function.
     * The only difference here from vanilla backprop is that we have
     * a 'reward_scaling' argument that makes the output error more/less
     * dramatic, so that we can adjust the weights proportionally to the
     * reward we want to provide. */
    static void backprop(NeuralNetwork nn, float[] targetProbs, float learningRate, float rewardScaling) {
        float[] outputDeltas = new float[NN_OUTPUT_SIZE];
        float[] hiddenDeltas = new float[NN_HIDDEN_SIZE];

        /* === STEP 1: Compute deltas === */

        /* Calculate output layer deltas:
         * Note what's going on here: we are technically using softmax
         * as output function and cross entropy as loss, but we never use
         * cross entropy in practice since we check the progresses in terms
         * of winning the game.
         *
         * Still calculating the deltas in the output as:
         *
         * output[i] - target[i]
         *
         * Is exactly what happens if you derivate the deltas with
         * softmax and cross entropy.
         *
         * LEARNING OPPORTUNITY: This is a well established and fundamental
         * result in neural networks, you may want to read more about it. */
        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            outputDeltas[i] =
                    (nn.outputs[i] - targetProbs[i]) * Math.abs(rewardScaling);
        }

        // Backpropagate error to hidden layer.
        for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
            float error = 0;
            for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
                error += outputDeltas[j] * nn.weightsHO[i * NN_OUTPUT_SIZE + j];
            }
            hiddenDeltas[i] = error * reluDerivative(nn.hidden[i]);
        }

        /* === STEP 2: Weights updating === */

        // Output layer weights and biases.
        for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
            for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
                nn.weightsHO[i * NN_OUTPUT_SIZE + j] -=
                        learningRate * outputDeltas[j] * nn.hidden[i];
            }
        }
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            nn.biasesO[j] -= learningRate * outputDeltas[j];
        }

        // Hidden layer weights and biases.
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
                nn.weightsIH[i * NN_HIDDEN_SIZE + j] -=
                        learningRate * hiddenDeltas[j] * nn.inputs[i];
            }
        }
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn.biasesH[j] -= learningRate * hiddenDeltas[j];
        }
    }

    /* Train the neural network based on game outcome.
     *
     * The move_history is just an integer array with the index of all the
     * moves. This function is designed so that you can specify if the
     * game was started by the move by the NN or human, but actually the
     * code always let the human move first. */
    static void learnFromGame(NeuralNetwork nn, int[] moveHistory, int numMoves, boolean nnMovesEven, char winner) {
        // Determine reward based on game outcome
        float reward;
        char nnSymbol = nnMovesEven ? 'O' : 'X';

        if (winner == 'T') {
            reward = 0.3f; // Small reward for draw
        } else if (winner == nnSymbol) {
            reward = 1.0f; // Large reward for win
        } else {
            reward = -2.0f; // Negative reward for loss
        }

        GameState state = new GameState();
        float[] targetProbs = new float[NN_OUTPUT_SIZE];

        // Process each move the neural network made.
        for (int moveIdx = 0; moveIdx < numMoves; moveIdx++) {
            // Skip if this wasn't a move by the neural network.
            if ((nnMovesEven && moveIdx % 2 != 1) ||
                    (!nnMovesEven && moveIdx % 2 != 0)) {
                continue;
            }

            // Recreate board state BEFORE this move was made.
            Arrays.fill(state.board, '.');
            state.currentPlayer = 0;
            for (int i = 0; i < moveIdx; i++) {
                char symbol = (i % 2 == 0) ? 'X' : 'O';
                state.board[moveHistory[i]] = symbol;
            }

            // Convert board to inputs and do forward pass.
            float[] inputs = new float[NN_INPUT_SIZE];
            boardToInputs(state, inputs);
            forwardPass(nn, inputs);

            /* The move that was actually made by the NN, that is
             * the one we want to reward (positively or negatively). */
            int move = moveHistory[moveIdx];

            /* Here we can't really implement temporal difference in the strict
             * reinforcement learning sense, since we don't have an easy way to
             * evaluate if the current situation is better or worse than the
             * previous state in the game.
             *
             * However "time related" we do something that is very effective in
             * this case: we scale the reward according to the move time, so that
             * later moves are more impacted (the game is less open to different
             * solutions as we go forward).
             *
             * We give a fixed 0.5 importance to all the moves plus
             * a 0.5 that depends on the move position.
             *
             * NOTE: this makes A LOT of difference. Experiment with different
             * values.
             *
             * LEARNING OPPORTUNITY: Temporal Difference in Reinforcement Learning
             * is a very important result, that was worth the Turing Award in
             * 2024 to Sutton and Barto. You may want to read about it. */
            float moveImportance = 0.5f + 0.5f * (float) moveIdx / (float) numMoves;
            float scaledReward = reward * moveImportance;

            /* Create target probability distribution:
             * let's start with the logits all set to 0. */
            Arrays.fill(targetProbs, 0);

            /* Set the target for the chosen move based on reward: */
            if (scaledReward >= 0) {
                /* For positive reward, set probability of the chosen move to
                 * 1, with all the rest set to 0. */
                targetProbs[move] = 1;
            } else {
                /* For negative reward, distribute probability to OTHER
                 * valid moves, which is conceptually the same as discouraging
                 * the move that we want to discourage. */
                int validMovesLeft = 9 - moveIdx - 1;
                float otherProb = 1.0f / validMovesLeft;
                for (int i = 0; i < 9; i++) {
                    if (state.board[i] == '.' && i != move) {
                        targetProbs[i] = otherProb;
                    }
                }
            }

            /* Call the generic backpropagation function, using
             * our target logits as target. */
            backprop(nn, targetProbs, LEARNING_RATE, scaledReward);
        }
    }

    /* Play one game of Tic Tac Toe against the neural network. */
    static void playGame(NeuralNetwork nn, Scanner scanner, Random random) {
        GameState state = new GameState();
        StringBuilder winner = new StringBuilder();
        int[] moveHistory = new int[9]; // Maximum 9 moves in a game.
        int numMoves = 0;

        System.out.println("Welcome to Tic Tac Toe! You are X, the computer is O.");
        System.out.println("Enter positions as numbers from 0 to 8 (see picture).");

        while (!checkGameOver(state, winner)) {
            displayBoard(state);

            if (state.currentPlayer == 0) {
                // Human turn.
                int move;
                char movec;
                System.out.print("Your move (0-8): ");
                movec = scanner.next().charAt(0);
                move = movec - '0'; // Turn character into number.

                // Check if move is valid.
                if (move < 0 || move > 8 || state.board[move] != '.') {
                    System.out.println("Invalid move! Try again.");
                    continue;
                }

                state.board[move] = 'X';
                moveHistory[numMoves++] = move;
            } else {
                // Computer's turn
                System.out.println("Computer's move:");
                int move = getComputerMove(state, nn, true);
                state.board[move] = 'O';
                System.out.printf("Computer placed O at position %d\n", move);
                moveHistory[numMoves++] = move;
            }

            state.currentPlayer = 1 - state.currentPlayer;
        }

        displayBoard(state);

        if (winner.toString().equals("X")) {
            System.out.println("You win!");
        } else if (winner.toString().equals("O")) {
            System.out.println("Computer wins!");
        } else {
            System.out.println("It's a tie!");
        }

        // Learn from this game
        learnFromGame(nn, moveHistory, numMoves, true, winner.charAt(0));
    }

    /* Get a random valid move, this is used for training
     * against a random opponent. Note: this function will loop forever
     * if the board is full, but here we want simple code. */
    static int getRandomMove(GameState state, Random random) {
        while (true) {
            int move = random.nextInt(9);
            if (state.board[move] != '.') continue;
            return move;
        }
    }

    /* Play a game against random moves and learn from it.
     *
     * This is a very simple Montecarlo Method applied to reinforcement
     * learning:
     *
     * 1. We play a complete random game (episode).
     * 2. We determine the reward based on the outcome of the game.
     * 3. We update the neural network in order to maximize future rewards.
     *
     * LEARNING OPPORTUNITY: while the code uses some Montecarlo-alike
     * technique, important results were recently obtained using
     * Montecarlo Tree Search (MCTS), where a tree structure repesents
     * potential future game states that are explored according to
     * some selection: you may want to learn about it. */
    static char playRandomGame(NeuralNetwork nn, int[] moveHistory, int[] numMoves, Random random) {
        GameState state = new GameState();
        StringBuilder winner = new StringBuilder();
        numMoves[0] = 0;

        while (!checkGameOver(state, winner)) {
            int move;

            if (state.currentPlayer == 0) { // Random player's turn (X)
                move = getRandomMove(state, random);
            } else { // Neural network's turn (O)
                move = getComputerMove(state, nn, false);
            }

            /* Make the move and store it: we need the moves sequence
             * during the learning stage. */
            char symbol = (state.currentPlayer == 0) ? 'X' : 'O';
            state.board[move] = symbol;
            moveHistory[numMoves[0]++] = move;

            // Switch player.
            state.currentPlayer = 1 - state.currentPlayer;
        }

        // Learn from this game - neural network is 'O' (even-numbered moves).
        learnFromGame(nn, moveHistory, numMoves[0], true, winner.charAt(0));
        return winner.charAt(0);
    }

    /* Train the neural network against random moves. */
    static void trainAgainstRandom(NeuralNetwork nn, int numGames, Random random) {
        int[] moveHistory = new int[9];
        int[] numMoves = new int[1];
        int wins = 0, losses = 0, ties = 0;

        System.out.printf("Training neural network against %d random games...\n", numGames);

        int playedGames = 0;
        for (int i = 0; i < numGames; i++) {
            char winner = playRandomGame(nn, moveHistory, numMoves, random);
            playedGames++;

            // Accumulate statistics that are provided to the user (it's fun).
            if (winner == 'O') {
                wins++; // Neural network won.
            } else if (winner == 'X') {
                losses++; // Random player won.
            } else {
                ties++; // Tie.
            }

            // Show progress every many games to avoid flooding the stdout.
            if ((i + 1) % 10000 == 0) {
                System.out.printf("Games: %d, Wins: %d (%.1f%%), " +
                                "Losses: %d (%.1f%%), Ties: %d (%.1f%%)\n",
                        i + 1, wins, (float) wins * 100 / playedGames,
                        losses, (float) losses * 100 / playedGames,
                        ties, (float) ties * 100 / playedGames);
                playedGames = 0;
                wins = 0;
                losses = 0;
                ties = 0;
            }
        }
        System.out.println("\nTraining complete!");
    }

    public static void main(String[] args) {
        int randomGames = 150000; // Fast and enough to play in a decent way.

        if (args.length > 0) randomGames = Integer.parseInt(args[0]);
        Random random = new Random();
        Scanner scanner = new Scanner(System.in);

        // Initialize neural network.
        NeuralNetwork nn = new NeuralNetwork();
        initNeuralNetwork(nn, random);

        // Train against random moves.
        if (randomGames > 0) trainAgainstRandom(nn, randomGames, random);

        // Play game with human and learn more.
        while (true) {
            System.out.print("Play again? (y/n): ");
            playGame(nn, scanner, random);
            String playAgain = scanner.next();
            if (!playAgain.equalsIgnoreCase("y")) break;
        }
        scanner.close();
    }
}