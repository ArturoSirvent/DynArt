import gradio as gr
import music21  
import fractions
import random
import random
from dataclasses import dataclass
import fractions
from music21 import midi, pitch
import copy
import matplotlib.pyplot as plt 
import numpy as np
from time import sleep

partitura=None
melody_obj=None
lista_partes=[]
lista_instrumentos={"Flute":music21.instrument.Flute()}


# @dataclass(frozen=True)
# class MelodyData:
#     """
#     A data class representing the data of a melody.

#     This class encapsulates the details of a melody including its notes, total
#     duration, and the number of bars. The notes are represented as a list of
#     tuples, with each tuple containing a pitch and its duration. The total
#     duration and the number of bars are computed based on the notes provided.

#     Attributes:
#         notes (list of tuples): List of tuples representing the melody's notes.
#             Each tuple is in the format (pitch, duration).
#         duration (int): Total duration of the melody, computed from notes.
#         number_of_bars (int): Total number of bars in the melody, computed from
#             the duration assuming a 4/4 time signature.

#     Methods:
#         __post_init__: A method called after the data class initialization to
#             calculate and set the duration and number of bars based on the
#             provided notes.
#     """

#     notes: list
#     duration: int = None  # Computed attribute
#     number_of_bars: int = None  # Computed attribute

#     def __post_init__(self):
#         object.__setattr__(
#             self, "duration", sum(duration for _, duration in self.notes)
#         )
#         object.__setattr__(self, "number_of_bars", int(self.duration // 2))


#example with all in it 

class MelodyData:
    """
    A data class representing the data of a melody.

    Attributes:
        notes (list of tuples): List of tuples representing the melody's notes.
        duration (int): Total duration of the melody, computed from notes.
        number_of_bars (int): Total number of bars in the melody.
        time_signature (music21.meter.TimeSignature): Time signature of the melody.
        scale (str): The scale of the melody.

    Methods:
        __post_init__: Calculates and sets the duration and number of bars.
        load_from_midi: Loads melody data from a MIDI file.
    """

    def __init__(self, file_path=None):
        self.notes: list = []
        self.duration: float = 0
        self.number_of_bars: int = 0
        self.time_signature: music21.meter.TimeSignature = None
        self.midi_data: music21.stream.Score  = None
        self.key: music21.key.Key = None
        self.tempo: music21.tempo.MetronomeMark=None
        self.durations_array:np.ndarray =[]
        self.ventana = 6  # Tamaño de la ventana para la mediana móvil
        self.alegro_points=[]
        #musical info
        self.chord_mappings_index = {
            "I": [0,4,7],
            "ii": [2,5,9],
            "iii": [4,7,11],
            "IV": [5,9,0],
            "V": [7,11,2],
            "vi": [9, 0, 4],
            "vii-dim": [11, 2, 5]
        }
        self.chord_names=list(self.chord_mappings_index.keys())
        self.chord_mappings={}
        self.escala_cromatica = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] 
        self.our_escala = []
        self.note_measure_correspondance=[]
        self.mapeo_bemoles = {
            "Db": "C#",
            "Eb": "D#",
            "Fb": "E",  # Fb es en realidad E
            "Gb": "F#",
            "Ab": "G#",
            "Bb": "A#",
            "Cb": "B"   # Cb es en realidad B
            }

        if file_path:
            self.load_from_midi(file_path)



    def __post_init__(self):
        self.duration = sum(duration for _, duration, _ in self.notes)
        if self.time_signature:
            self.number_of_bars = int(self.duration / (self.time_signature.numerator ))
    def mediana_movil(self,arr):
        # La longitud de la mediana móvil será menor que la del array original
        # debido a los bordes donde la ventana no se ajusta completamente.
        num_elementos = len(arr) - self.ventana + 1
        mediana_movil = np.zeros(len(arr))

        for i in range(num_elementos):
            mediana_movil[i] = np.median(arr[i:(i+self.ventana)])

        return mediana_movil
    
    @staticmethod
    def gaussian(x, mu, sig):
        return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )

    def load_from_midi(self, file_path):

        #populate class
        self.midi_data = music21.converter.parse(file_path)
        self.time_signature = self.midi_data.recurse().getElementsByClass('TimeSignature')[0]
        # self.key = self.midi_data.analyze('key') #esto será para sacarlo analizandolo no la Key Signature
        try:
            self.key = self.midi_data.flat.getElementsByClass(music21.key.Key)[0]
        except:
            self.key = self.midi_data.flat.getElementsByClass(music21.key.KeySignature)[0].asKey()
        self.tempo=self.midi_data.recurse().getElementsByClass(music21.tempo.MetronomeMark)[0]

        #retrive melody
        parts = [part.flat for part in self.midi_data.parts]
        for n in parts[0]:
            try:
                if n.isNote:
                    nombre_nota = n.pitch.name
                    octava= n.pitch.octave
                    if nombre_nota in self.mapeo_bemoles.keys():
                        nombre_nota=self.mapeo_bemoles[nombre_nota]
                    duracion = n.duration.quarterLength
                elif n.isRest:
                    nombre_nota = "Rest"
                    octava= 5
                    duracion = n.duration.quarterLength
                elif n.isChord:
                    nombre_nota = [i.pitch.name for i in n.notes]
                    octava= [i.pitch.octave for i in n.notes]
                    if any([i in self.mapeo_bemoles.keys() for i in nombre_nota]):
                        nombre_nota=[self.mapeo_bemoles[i] for i in nombre_nota]
                    octava=octava[0]
                    nombre_nota=nombre_nota[0]#esto es algo temporal para evitar tener problemas con las listas
                    duracion = n.notes[0].duration.quarterLength#[i.duration.quarterLength for i in nota.notes]
                else:
                    print(f"error con {n}")
                    continue

                if isinstance(duracion,fractions.Fraction):
                    duracion=float(duracion)
                self.note_measure_correspondance.append(n.measureNumber)

                note_repr = (nombre_nota, duracion,octava)
                self.notes.append(note_repr)
                aux_duration= duracion#-1* duracion if n.isRest else duracion
                self.durations_array.append(aux_duration)
            except:
                print(f"error con {n}")
        if self.durations_array:        
            self.durations_array=np.array(self.durations_array)

        # Restar uno a cada elemento de la lista
        self.note_measure_correspondance = [x - 1 for x in self.note_measure_correspondance]

        # Crear un diccionario para agrupar las posiciones
        aux_note_measure_correspondance_diccionario = {}
        for idx, valor in enumerate(self.note_measure_correspondance):
            if valor in aux_note_measure_correspondance_diccionario:
                aux_note_measure_correspondance_diccionario[valor].append(idx)
            else:
                aux_note_measure_correspondance_diccionario[valor] = [idx]

        self.note_measure_correspondance=aux_note_measure_correspondance_diccionario

        # Ejemplo de uso
        self.alegro_points = self.mediana_movil(self.durations_array)
        #algunos puntos son 0 porque la mediana al final no tenemos elementos para calcularla.
        #calculamos la media de el resto y asignamos unos pesos probabilisticos para elegir segun esta diferencia, la probabilidad de añadir un parte rapida
        #media_aux=self.alegro_points[self.alegro_points!=0].mean()
        bool_small_duration=np.where(self.durations_array<self.alegro_points[self.alegro_points!=0].mean(),1,0)
        gaus_aux=self.gaussian(np.linspace(-2,2,self.alegro_points.size),0,1)
        self.positions_allegro=[]
        for i in np.arange(self.alegro_points.size):
            if random.random()<gaus_aux[i]:
                if bool_small_duration[i]>0:
                    self.positions_allegro.append(i)
        
        # scale data
        #len_escala=len(self.escala_cromatica)
        nota_tonica = self.key.tonicPitchNameWithCase
        pos_0 = self.escala_cromatica.index(nota_tonica)
        self.our_escala=self.escala_cromatica[pos_0:]+self.escala_cromatica[:pos_0]
        for i_key,i_value in self.chord_mappings_index.items():
            self.chord_mappings[i_key]=[self.our_escala[i] for i in i_value]  
            
        # if n.isNote:
        #     note_repr = (n.pitch.midi, n.duration.quarterLength)
        #     self.notes.append(note_repr)
        # elif n.isRest:
        #     rest_repr = ('rest', n.duration.quarterLength)
        #     self.notes.append(rest_repr)

        self.__post_init__()

    # def midi_note_to_name_dur(self, note):
    #     if note.isNote:
    #         return (note.pitch.midi, note.duration.quarterLength)
    #     elif note.isRest:
    #         return ('rest', note.duration.quarterLength)
    #     return None



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
            random.choice(self.melody_data.chord_names)
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
            chord_sequence[mutation_index] = random.choice(self.melody_data.chord_names)
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
        self, melody_data, weights, preferred_transitions):
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
        self.chord_mappings = melody_data.chord_mappings
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
                pitch, duration, _ = self.melody_data.notes[melody_index]
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
        if chord_sequence[0] in ["I"]:
            score += 1
        if chord_sequence[-1] in ["I"]:
            score += 1
        if "IV" in chord_sequence and "V" in chord_sequence:
            score += 1
        return score / 3


# Ideas de regla:  
# 1: En cada frase, acordes con funcion tonal. I, vi, IV  
# 2: Transiciones ponderadas entre notas (ahora todas pesan lo mismo )   
# 3: Cada 4 compases, chord Variability. WRONG. Wider range , maybe 20 beats. 
# 4: Permitir escalas no tipicos como La mayor.  
# 5: sería idea indicar con el movimiento que un acorde sean varias notas (fragmentacion del acorde). Y si no es con la interactividad, añadir una 
# 6: opcion, de añadir varios grupos de instrumentos y subir o bajarles conjuntamente.  

def midi_note_to_name_dur(nota):
    try:
        if nota.isNote:
            nombre_nota = pitch.Pitch(nota.pitch).nameWithOctave
            duracion = nota.duration.quarterLength
        elif nota.isRest:
            nombre_nota = "Rest"
            duracion = nota.duration.quarterLength
        elif nota.isChord:
            nombre_nota = [pitch.Pitch(i.pitch).nameWithOctave for i in nota.notes]
            duracion = nota.notes[0].duration.quarterLength#[i.duration.quarterLength for i in nota.notes]
        else:
            print(f"error con {nota}")
        if isinstance(duracion,fractions.Fraction):
            duracion=float(duracion)
        return (nombre_nota, duracion)
    except:
        print(f"error con {nota}")
        return None
    

preferred_transitions = {
    "I": ["V", "vi", "IV"],
    "ii": ["V", "vi"],
    "iii": ["vi", "F", "C"],
    "IV": ["I", "V","ii","iii"],
    "V": ["vi", "I","IV"],
    "vi": ["ii", "iii", "IV"],
    "vii-dim": ["I"]
}

# cuerda
conf_harm_cuerda={"Channel_1":{"name": "Flute_Melody" , "instrument":music21.instrument.Flute(),"level":"Melody" , "octave":None,"repetition":True },
           "Channel_2":{"name": "Soprano" , "instrument":music21.instrument.Violin(),"level":"Melody" , "octave":5 ,"repetition":True},
           "Channel_3":{"name": "Alto" , "instrument":music21.instrument.Violin(),"level":[1] , "octave": 5 ,"repetition":True},
           "Channel_4":{"name": "Tenor" , "instrument":music21.instrument.Viola(),"level":[2] , "octave": 4 ,"repetition":True},
           "Channel_5":{"name": "Chello" , "instrument":music21.instrument.Violoncello(),"level":[0] , "octave":3 ,"repetition":False},
           "Channel_6":{"name": "DoubleBass" , "instrument":music21.instrument.Contrabass(),"level":[0] , "octave":3 ,"repetition":False}
           }

# viento metal
conf_harm_viento_metal={"Channel_7":{"name": "Tuba" , "instrument":music21.instrument.Tuba(),"level":[0] , "octave":3 ,"repetition":True},
           "Channel_8":{"name": "Trombone" , "instrument":music21.instrument.Trombone(),"level":[2] , "octave":3 ,"repetition":False},
           "Channel_9":{"name": "FrenchHorn" , "instrument":music21.instrument.Horn(),"level":[1] , "octave": 5 ,"repetition":False}
           }


# viento madera
conf_harm_viento_madera={"Channel_10":{"name": "Fagot" , "instrument":music21.instrument.EnglishHorn(),"level":[0] , "octave":3 ,"repetition":False},
           "Channel_11":{"name": "Clarinet" , "instrument":music21.instrument.Clarinet(),"level":[2] , "octave":3 ,"repetition":True},
           "Channel_12":{"name": "Flute" , "instrument":music21.instrument.Flute(),"level":[1] , "octave": 4 ,"repetition":True}
           }


def create_score2(melody, chord_sequence, conf_harm):
    chord_mappings=melody.chord_mappings
    indices_alegro=melody.positions_allegro
    score = music21.stream.Score()
    duration_per_section = melody.time_signature.numerator
    number_notes_in_fast_measures=2
    small_duration_per_section=duration_per_section/number_notes_in_fast_measures
    score.append(melody.tempo)
    for channel, config in conf_harm.items():
        part = music21.stream.Part()
        part.append(config["instrument"])
        part.append(copy.deepcopy(melody.time_signature))


        if config["level"] == "Melody":
            for note_name, duration,octave in melody.notes:
                if config["octave"] is not None:
                    octave=config["octave"]
                if not isinstance(note_name, list):
                    if note_name == "Rest":
                        music_note = music21.note.Rest(quarterLength=duration)
                    else:
                        music_note = music21.note.Note(note_name, 
                                                       quarterLength=duration, 
                                                       octave=octave)
                else:
                    music_note = music21.chord.Chord(note_name, 
                                                     quarterLength=duration, 
                                                     octave=octave)
                part.append(music_note)
        else:
            current_duration = 0
            for i,chord_name in enumerate(chord_sequence):
                if (any([j in melody.note_measure_correspondance[i] for j in melody.positions_allegro]))&(config["repetition"]):
                    for _ in range(number_notes_in_fast_measures):
                        chord_notes_list = [chord_mappings.get(chord_name, [])[i] for i in config["level"]]
                        chord_notes = music21.chord.Chord([music21.note.Note(i, octave=config["octave"]) for i in chord_notes_list], 
                                                        quarterLength=small_duration_per_section)
                        chord_notes.offset = current_duration
                        part.append(chord_notes)
                        current_duration += small_duration_per_section
                else:
                    chord_notes_list = [chord_mappings.get(chord_name, [])[i] for i in config["level"]]
                    chord_notes = music21.chord.Chord([music21.note.Note(i, octave=config["octave"]) for i in chord_notes_list], 
                                                    quarterLength=duration_per_section)
                    chord_notes.offset = current_duration
                    part.append(chord_notes)
                    current_duration += duration_per_section

        score.append(part)

    return score



def cargar_archivo(file_name='./data-MIDI/carmen_flue.musicxml'):
    global melody
    melody=MelodyData(file_path=file_name)
    instrumentos=[i.instrumentName for i in melody.midi_data.getInstruments().elements]
    print(instrumentos)

    return gr.CheckboxGroup(label="Select instruments",choices=instrumentos,interactive=True)

def seleccion_instrumento(seleccion):
    global melody
    tmp_partitura=music21.stream.Score()
    partitura_by_instrument=music21.instrument.partitionByInstrument(melody.midi_data)
    for i in partitura_by_instrument:
        inst=i.getInstrument().instrumentName
        if inst in seleccion:
            tmp_partitura.append(i.flat)
    return "Instruments loaded!"

def entrenar(peso1, peso2, peso3, peso4, peso5, poblacion, epocas):
    global melody, preferred_transitions,conf_harm_cuerda,conf_harm_viento_madera,conf_harm_viento_metal
    weights = {
        "chord_melody_congruence": peso1,  #0.6,
        "chord_variety": peso2, #0.23,
        "harmonic_flow": peso3,#0.05,
        "functional_harmony": peso4#0.1
    }

    print(melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody,
        weights=weights,
        preferred_transitions=preferred_transitions,
    )

    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody,
        population_size=int(poblacion),
        mutation_rate=peso5,
        fitness_evaluator=fitness_evaluator,
    )

    generated_chords = harmonizer.generate(generations=epocas)

    # Render to music21 score and show it
    conf_harm_cuerda.update(conf_harm_viento_madera)
    conf_harm_cuerda.update(conf_harm_viento_metal)
    music21_score = create_score2(melody, generated_chords,conf_harm_cuerda)

    music21_score.write('midi', fp='./data-MIDI/result.mid')


    
    return f"Training completed: Population {poblacion}, Epochs {epocas}" #gr.Button.update(link="file/=" +"./data-MIDI/C4.mid")


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            archivo = gr.File(label="Load MIDI/XML file")
            example=gr.Examples([['./data-MIDI/carmen_flue.musicxml'],["./data-MIDI/C4.mid"]],inputs=[archivo])
            cargar_btn = gr.Button("Load")
        with gr.Column(scale=1):
            instrumentos = gr.CheckboxGroup(label="Upload file to see instruments", choices=[])
            seleccion_btn = gr.Button("Select")
            cargar_output = gr.Textbox()
            
    cargar_btn.click(cargar_archivo, inputs=archivo, outputs=instrumentos)
    seleccion_btn.click(seleccion_instrumento, inputs=[instrumentos], outputs=[cargar_output])


    with gr.Row():
        mostrar_btn = gr.Button("Mostrar")
        escuchar_btn = gr.Button("Escuchar")

    # mostrar_btn.click(mostrar_partitura, inputs=None, outputs=mostrar_output)
    # escuchar_btn.click(escuchar_partitura, inputs=None, outputs=escuchar_output)

    with gr.Row():
        with gr.Column(scale=1):
            peso1 = gr.Slider(minimum=0, maximum=1, label="Weight chord_melody_congruence")
            peso2 = gr.Slider(minimum=0, maximum=1, label="Weight chord_variety")
            peso3 = gr.Slider(minimum=0, maximum=1, label="Weight functional_harmony")
        with gr.Column(scale=1):
            peso4 = gr.Slider(minimum=0, maximum=1, label="Weight harmonic_flow")
            peso5 = gr.Slider(minimum=0, maximum=1,step=0.01, label="mutation rate")
        #opciones = gr.CheckboxGroup(choices=["Opción 1", "Opción 2", "Opción 3", "Opción 4", "Opción 5"], label="Opciones adicionales")
        poblacion = gr.Slider(minimum=2, maximum=1000,step=2 ,label="Population")
        epocas = gr.Slider(minimum=1, maximum=100, label="Epocs",step=1)

    entrenar_output = gr.Textbox(info="Summary of training")
    entrenar_btn = gr.Button("Train")
    download_button = gr.Button("Dowload",link="result.mid")

    entrenar_btn.click(entrenar, inputs=[peso1, peso2, peso3, peso4, peso5, poblacion, epocas], outputs=entrenar_output)


demo.launch(allowed_paths=["result.mid"])

