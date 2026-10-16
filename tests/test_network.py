"""Tests für das Feedforward-Netzwerk (handnachprüfbare Mathematik)."""
import numpy as np
import pytest

from mario_evolution.network import NeuralNetwork, relu, identity


class TestActivation:
    def test_relu_negative_zero_positive(self):
        """ReLU: f(z)=max(0,z) - negativ->0, null->0, positiv->unverändert."""
        z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        assert np.allclose(relu(z), [0.0, 0.0, 0.0, 0.5, 3.0])

    def test_relu_rejects_nan(self):
        with pytest.raises(ValueError):
            relu(np.array([1.0, np.nan]))

    def test_identity(self):
        assert np.allclose(identity(np.array([-1.0, 2.0])), [-1.0, 2.0])


class TestManualNeuron:
    def test_single_layer_forward_by_hand(self):
        """Kleinstes Netz [1,1,1]: z = w*x+b, a = relu(z) - von Hand prüfbar."""
        net = NeuralNetwork(
            layer_sizes=[1, 1, 1],
            weights=[np.array([[2.0]]), np.array([[2.0]])],
            biases=[np.array([1.0]), np.array([1.0])],
        )
        # hidden: z = 2*3 + 1 = 7 -> relu(7) = 7
        # output: z = 2*7 + 1 = 15 -> identity(15) = 15
        assert net.forward(np.array([3.0]))[0] == pytest.approx(15.0)

    def test_negative_input_is_clamped_by_relu(self):
        """Negatives z wird von ReLU auf 0 geklemmt (Ergebnis der nächsten Schicht: b)."""
        net = NeuralNetwork(
            layer_sizes=[1, 1, 1],
            weights=[np.array([[2.0]]), np.array([[2.0]])],
            biases=[np.array([1.0]), np.array([1.0])],
        )
        # hidden: z = 2*(-2)+1 = -3 -> relu = 0
        # output: z = 2*0 + 1 = 1
        assert net.forward(np.array([-2.0]))[0] == pytest.approx(1.0)

    def test_two_hidden_neurons_manual_matrix(self):
        """Matrixform: W (2x1) @ x (1) + b -> 2 versteckte Werte."""
        net = NeuralNetwork(
            layer_sizes=[1, 2, 1],
            weights=[np.array([[1.0], [-1.0]]), np.array([[1.0, 1.0]])],
            biases=[np.array([0.0, 0.5]), np.array([0.0])],
        )
        # hidden: z1 = 1*4+0 = 4 -> 4; z2 = -1*4+0.5 = -3.5 -> relu = 0
        # output: z = 1*4 + 1*0 = 4
        assert net.forward(np.array([4.0]))[0] == pytest.approx(4.0)


class TestShapesAndValidation:
    def test_weight_shapes(self):
        net = NeuralNetwork([5, 3, 2], rng=np.random.default_rng(0))
        assert net.weights[0].shape == (3, 5)
        assert net.weights[1].shape == (2, 3)
        assert net.biases[0].shape == (3,)
        assert net.biases[1].shape == (2,)

    def test_wrong_input_size_raises(self):
        net = NeuralNetwork([5, 3, 2], rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="Eingabewerte"):
            net.forward(np.zeros(4))

    def test_nan_input_raises(self):
        net = NeuralNetwork([2, 2, 2], rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="endlich"):
            net.forward(np.array([np.nan, 0.0]))

    def test_invalid_layer_sizes(self):
        with pytest.raises(ValueError):
            NeuralNetwork([4, 2])          # zu wenige Schichten
        with pytest.raises(ValueError):
            NeuralNetwork([4, 0, 2])       # leere Schicht
        with pytest.raises(TypeError):
            NeuralNetwork([4, 2.5, 2])     # keine int-Größe

    def test_wrong_parameter_shapes_rejected(self):
        with pytest.raises(ValueError, match="Form"):
            NeuralNetwork([2, 2, 2], weights=[np.zeros((3, 2)), np.zeros((2, 3))],
                          biases=[np.zeros(2), np.zeros(2)])

    def test_nan_parameters_rejected(self):
        w = np.array([[np.nan, 0.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="NaN"):
            NeuralNetwork([2, 2, 2], weights=[w, np.zeros((2, 2))],
                          biases=[np.zeros(2), np.zeros(2)])


class TestReproducibility:
    def test_same_seed_same_parameters(self):
        n1 = NeuralNetwork([3, 4, 2], rng=np.random.default_rng(42))
        n2 = NeuralNetwork([3, 4, 2], rng=np.random.default_rng(42))
        assert np.allclose(n1.genome(), n2.genome())

    def test_different_seed_different_parameters(self):
        n1 = NeuralNetwork([3, 4, 2], rng=np.random.default_rng(42))
        n2 = NeuralNetwork([3, 4, 2], rng=np.random.default_rng(43))
        assert not np.allclose(n1.genome(), n2.genome())


class TestGenome:
    def test_genome_roundtrip_preserves_outputs(self):
        net = NeuralNetwork([5, 4, 2], rng=np.random.default_rng(7))
        clone = net.from_genome(net.genome())
        x = np.random.default_rng(1).random(5)
        assert np.allclose(net.forward(x), clone.forward(x))

    def test_genome_size(self):
        net = NeuralNetwork([5, 4, 2], rng=np.random.default_rng(7))
        # Gewichte: 4*5 + 2*4 = 28; Bias: 4 + 2 = 6; Summe 34
        assert net.genome_size() == 34
        assert net.genome().shape == (34,)

    def test_wrong_genome_length_rejected(self):
        net = NeuralNetwork([5, 4, 2], rng=np.random.default_rng(7))
        with pytest.raises(ValueError, match="Genom"):
            net.from_genome(np.zeros(10))

    def test_describe(self):
        assert NeuralNetwork([5, 4, 2], rng=np.random.default_rng(0)).describe() == "5-4-2"
