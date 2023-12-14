import gradio as gr
import music21  
import fractions
from dataclasses import dataclass
import random

partitura=None
lista_partes=[]
lista_instrumentos={"Flute":music21.instrument.Flute()}


@dataclass(frozen=True)
class MelodyData:
    """
    A data class representing the data of a melody.

    This class encapsulates the details of a melody including its notes, total
    duration, and the number of bars. The notes are represented as a list of
    tuples, with each tuple containing a pitch and its duration. The total
    duration and the number of bars are computed based on the notes provided.

    Attributes:
        notes (list of tuples): List of tuples representing the melody's notes.
            Each tuple is in the format (pitch, duration).
        duration (int): Total duration of the melody, computed from notes.
        number_of_bars (int): Total number of bars in the melody, computed from
            the duration assuming a 4/4 time signature.

    Methods:
        __post_init__: A method called after the data class initialization to
            calculate and set the duration and number of bars based on the
            provided notes.
    """

    notes: list
    duration: int = None  # Computed attribute
    number_of_bars: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_bars", int(self.duration // 2))


class GeneticMelodyHarmonizer:
    """
    Generates chord accompaniments for a given melody using a genetic algorithm.
    It evolves a population of chord sequences to find one that best fits the
    melody based on a fitness function.

    Attributes:
        melody_data (MusicData): Data containing melody information.
        chords (list): Available chords for generating sequences.
        population_size (int): Size of the chord sequence population.
        mutation_rate (float): Probability of mutation in the genetic algorithm.
        fitness_evaluator (FitnessEvaluator): Instance used to assess fitness.
    """

    def __init__(
        self,
        melody_data,
        chords,
        population_size,
        mutation_rate,
        fitness_evaluator,
    ):
        """
        Initializes the generator with melody data, chords, population size,
        mutation rate, and a fitness evaluator.

        Parameters:
            melody_data (MusicData): Melody information.
            chords (list): Available chords.
            population_size (int): Size of population in the algorithm.
            mutation_rate (float): Mutation probability per chord.
            fitness_evaluator (FitnessEvaluator): Evaluator for chord fitness.
        """
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = int(population_size)
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
        """
        Generates a chord sequence that harmonizes a melody using a genetic
        algorithm.

        Parameters:
            generations (int): Number of generations for evolution.

        Returns:
            best_chord_sequence (list): Harmonization with the highest fitness
                found in the last generation.
        """
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = (
            self.fitness_evaluator.get_chord_sequence_with_highest_fitness(
                self._population
            )
        )
        return best_chord_sequence

    def _initialise_population(self):
        """
        Initializes population with random chord sequences.

        Returns:
            list: List of randomly generated chord sequences.
        """
        return [
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        """
        Generate a random chord sequence with as many chords as the numbers
        of bars in the melody.

        Returns:
            list: List of randomly generated chords.
        """
        return [
            random.choice(self.chords)
            for _ in range(self.melody_data.number_of_bars)
        ]

    def _select_parents(self):
        """
        Selects parent sequences for breeding based on fitness.

        Returns:
            list: Selected parent chord sequences.
        """
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        """
        Generates a new population of chord sequences from the provided parents.

        This method creates a new generation of chord sequences using crossover
        and mutation operations. For each pair of parent chord sequences,
        it generates two children. Each child is the result of a crossover
        operation between the pair of parents, followed by a potential
        mutation. The new population is formed by collecting all these
        children.

        The method ensures that the new population size is equal to the
        predefined population size of the generator. It processes parents in
        pairs, and for each pair, two children are generated.

        Parameters:
            parents (list): A list of parent chord sequences from which to
                generate the new population.

        Returns:
            list: A new population of chord sequences, generated from the
                parents.

        Note:
            This method assumes an even population size and that the number of
            parents is equal to the predefined population size.
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Combines two parent sequences into a new child sequence using one-point
        crossover.

        Parameters:
            parent1 (list): First parent chord sequence.
            parent2 (list): Second parent chord sequence.

        Returns:
            list: Resulting child chord sequence.
        """
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence):
        """
        Mutates a chord in the sequence based on mutation rate.

        Parameters:
            chord_sequence (list): Chord sequence to mutate.

        Returns:
            list: Mutated chord sequence.
        """
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence
    

class FitnessEvaluator:
    """
    Evaluates the fitness of a chord sequence based on various musical criteria.

    Attributes:
        melody (list): List of tuples representing notes as (pitch, duration).
        chords (dict): Dictionary of chords with their corresponding notes.
        weights (dict): Weights for different fitness evaluation functions.
        preferred_transitions (dict): Preferred chord transitions.
    """

    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ):
        """
        Initialize the FitnessEvaluator with melody, chords, weights, and
        preferred transitions.

        Parameters:
            melody_data (MelodyData): Melody information.
            chord_mappings (dict): Available chords mapped to their notes.
            weights (dict): Weights for each fitness evaluation function.
            preferred_transitions (dict): Preferred chord transitions.
        """
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        """
        Returns the chord sequence with the highest fitness score.

        Parameters:
            chord_sequences (list): List of chord sequences to evaluate.

        Returns:
            list: Chord sequence with the highest fitness score.
        """
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        """
        Evaluate the fitness of a given chord sequence.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: The overall fitness score of the chord sequence.
        """
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        """
        Calculates the congruence between the chord sequence and the melody.
        This function assesses how well each chord in the sequence aligns
        with the corresponding segment of the melody. The alignment is
        measured by checking if the notes in the melody are present in the
        chords being played at the same time, rewarding sequences where the
        melody notes fit well with the chords.

        Parameters:
            chord_sequence (list): A list of chords to be evaluated against the
                melody.

        Returns:
            float: A score representing the degree of congruence between the
                chord sequence and the melody, normalized by the melody's
                duration.
        """
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 2 and melody_index < len(
                self.melody_data.notes
            ):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence):
        """
        Evaluates the diversity of chords used in the sequence. This function
        calculates a score based on the number of unique chords present in the
        sequence compared to the total available chords. Higher variety in the
        chord sequence results in a higher score, promoting musical
        complexity and interest.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score representing the variety of chords in the
                sequence relative to the total number of available chords.
        """
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        """
        Assesses the harmonic flow of the chord sequence by examining the
        transitions between successive chords. This function scores the
        sequence based on how frequently the chord transitions align with
        predefined preferred transitions. Smooth and musically pleasant
        transitions result in a higher score.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score based on the frequency of preferred chord
                transitions in the sequence.
        """
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        """
        Evaluates the chord sequence based on principles of functional harmony.
        This function checks for the presence of key harmonic functions such as
        the tonic at the beginning and end of the sequence and the presence of
        subdominant and dominant chords. Adherence to these harmonic
        conventions is rewarded with a higher score.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A score representing the extent to which the sequence
                adheres to traditional functional harmony, normalized by
                the number of checks performed.
        """
        score = 0
        if chord_sequence[0] in ["C", "Am"]:
            score += 1
        if chord_sequence[-1] in ["C"]:
            score += 1
        if "F" in chord_sequence and "G" in chord_sequence:
            score += 1
        return score / 3





def midi_note_to_name_dur(nota):
    try:
        if nota.isNote:
            nombre_nota = music21.pitch.Pitch(nota.pitch).nameWithOctave
            duracion = nota.duration.quarterLength
        elif nota.isRest:
            nombre_nota = "Rest"
            duracion = nota.duration.quarterLength
        elif nota.isChord:
            nombre_nota = [music21.pitch.Pitch(i.pitch).nameWithOctave for i in nota.notes]
            duracion = nota.notes[0].duration.quarterLength#[i.duration.quarterLength for i in nota.notes]
        else:
            pass#print(f"error con {nota}")
        if isinstance(duracion,fractions.Fraction):
            duracion=float(duracion)
        return (nombre_nota, duracion)
    except:
        pass#print(f"error con {nota}")
        return None

def cargar_archivo(file_name='./data-MIDI/carmen_flue.musicxml'):
    global partitura
    partitura_load=music21.converter.parse(file_name)
    partitura = partitura_load
    instrumentos=[i.instrumentName for i in partitura.getInstruments().elements]
    print(instrumentos)

    return gr.CheckboxGroup(label="Select instruments",choices=instrumentos,interactive=True)

def seleccion_instrumento(seleccion):
    global partitura, lista_partes
    tmp_partitura=music21.stream.Score()
    partitura_by_instrument=music21.instrument.partitionByInstrument(partitura)
    for i in partitura_by_instrument:
        inst=i.getInstrument().instrumentName
        if inst in seleccion:
            tmp_partitura.append(i.flat)

    partitura=tmp_partitura

    lista_partes=[]

    for parts in partitura:
        tuplas_notas_aux=[]
        for n in parts:
            aux1=midi_note_to_name_dur(n)

            if aux1 is not None:
                tuplas_notas_aux.append(aux1)

        lista_partes.append(tuplas_notas_aux)

          

def mostrar_partitura():
    # Lógica para mostrar la partitura
    return "Mostrando partitura"

def escuchar_partitura():
    # Lógica para escuchar la partitura
    return "Reproduciendo partitura"

def entrenar(peso1, peso2, peso3, peso4, peso5, poblacion, epocas):
    global lista_partes
    print(peso1, peso2, peso3, peso4, peso5, poblacion, epocas)
    weights = {
        "chord_melody_congruence": peso1,  #0.6,
        "chord_variety": peso2, #0.23,
        "harmonic_flow": peso3,#0.05,
        "functional_harmony": peso4#0.1
    }

    # DoubleBass (primera nota y octava 3) , Chello (primera nota y octava 3) , Viola (Tercera nota y octava 4 ) , Violin (segunda nota en octava 5)
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    preferred_transitions = {
        "C": ["G", "Am", "F"],
        "Dm": ["G", "Am"],
        "Em": ["Am", "F", "C"],
        "F": ["C", "G"],
        "G": ["Am", "C"],
        "Am": ["Dm", "Em", "F", "C"],
        "Bdim": ["F", "Am"]
    }
    melody=MelodyData(lista_partes[int(peso5)])
    print(melody.number_of_bars,melody.duration)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody,
        chords=list(chord_mappings.keys()),
        population_size=int(poblacion),
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator,
    )
    generated_chords = harmonizer.generate(generations=epocas)
    
    return f"Entrenamiento completado: Población {poblacion}, Épocas {epocas}"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            archivo = gr.File(label="Cargar archivo MIDI/XML")
            cargar_btn = gr.Button("Cargar")
        with gr.Column(scale=1):
            instrumentos = gr.CheckboxGroup(label="Upload file to see instruments", choices=[])
            seleccion_btn = gr.Button("Seleccionar")
            cargar_output = gr.Textbox()
            
    cargar_btn.click(cargar_archivo, inputs=archivo, outputs=instrumentos)
    seleccion_btn.click(seleccion_instrumento, inputs=[instrumentos], outputs=[])


    with gr.Row():
        mostrar_output = gr.Textbox()
        escuchar_output = gr.Textbox()
        mostrar_btn = gr.Button("Mostrar")
        escuchar_btn = gr.Button("Escuchar")

    mostrar_btn.click(mostrar_partitura, inputs=None, outputs=mostrar_output)
    escuchar_btn.click(escuchar_partitura, inputs=None, outputs=escuchar_output)

    with gr.Row():
        with gr.Column(scale=1):
            peso1 = gr.Slider(minimum=0, maximum=1, label="Weight 1")
            peso2 = gr.Slider(minimum=0, maximum=1, label="Weight 2")
            peso3 = gr.Slider(minimum=0, maximum=1, label="Weight 3")
        with gr.Column(scale=1):
            peso4 = gr.Slider(minimum=0, maximum=1, label="Weight 4")
            peso5 = gr.Slider(minimum=0, maximum=3,step=1, label="index")
        opciones = gr.CheckboxGroup(choices=["Opción 1", "Opción 2", "Opción 3", "Opción 4", "Opción 5"], label="Opciones adicionales")
        poblacion = gr.Slider(minimum=1, maximum=1000, label="Población",step=1)
        epocas = gr.Slider(minimum=1, maximum=100, label="Épocas",step=1)

    entrenar_output = gr.Textbox()
    entrenar_btn = gr.Button("Entrenar")
    entrenar_btn.click(entrenar, inputs=[peso1, peso2, peso3, peso4, peso5, poblacion, epocas], outputs=entrenar_output)

    with gr.Row():
        # Panel para mostrar resultados después del entrenamiento
        resultados = gr.Textbox(label="Resultados del entrenamiento")

demo.launch()


# import gradio as gr

# def actualizar_opciones():
#     # Esta función devuelve una lista de opciones para el CheckboxGroup
#     nuevas_opciones = ["1","2"]
#     return nuevas_opciones#gr.CheckboxGroup(choices=nuevas_opciones,interactive=True)

# def print_selections(inputs):
#     print(inputs)

# with gr.Blocks() as demo:
#     with gr.Row():
#         boton_actualizar = gr.Button("Actualizar Opciones")
#         checkbox_group = gr.CheckboxGroup(label="Elige opciones", choices=["a"])

#     boton_actualizar.click(fn=actualizar_opciones, inputs=[], outputs=[checkbox_group])
#     checkbox_group.change(print_selections,inputs=[checkbox_group])

# demo.launch()

