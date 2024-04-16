import numpy as np
from deap import gp, creator, tools
from tpot import gp_types
from datetime import datetime
from settings import tpot_settings


def split_scores(scores_list):
    """Split up scores_list into overall scores and individual scores per target."""
    overall_scores_list = [score[:1] for score in scores_list]
    overall_scores = np.array(overall_scores_list)[:, 0]
    overall_cv_score = np.nanmean(overall_scores)

    ind_scores_list = [score[1] for score in scores_list]
    ind_scores_array = np.concatenate([ind_scores_list], axis=1)
    ind_cv_scores = np.average(ind_scores_array, axis=0)

    return overall_cv_score, ind_cv_scores.tolist()

def finddupes(pop):
    """Find duplicates in a population and return a tuple with information about the dupe.
    """

    dupes = []
    SetOfIndividuals = set()

    for ind in pop:
        ind_str = str(ind)
        if ind_str in SetOfIndividuals:
            # Dupe found
            dupe = (tpot_settings.random_state, ind.statistics['generation'], ind_str, 'Duplicate(CP)', datetime.now())
            dupes.append(dupe)
        else:
            SetOfIndividuals.add(ind_str)

    return dupes


def dicttohof(hof_dict, tpot_operators):
    """Generate a Hall of Fame (Pareto Front) by using the corresponding dict.

    :param hof_dict: Dict with information about the saved Hall of Fame.
    :param tpot_operators: List of <class 'tpot.operator_utils...'> used to assign operators.

    :return hof: Hall of Fame
    """

    def pareto_eq(ind1, ind2):
        """Copied from tpot/base.py
        Used to generate Pareto_Front
        """
        return np.allclose(ind1.fitness.values, ind2.fitness.values)

    hof = tools.ParetoFront(similar=pareto_eq)

    # reassign attributes of hall of fame
    # items
    for ind_dict in hof_dict['items']:
        ind = dicttoind(ind_dict, tpot_operators)
        hof.items.append(ind)

    # keys
    for ind in reversed(hof.items):
        hof.keys.append(ind.fitness)

    hof.maxsize = hof_dict['maxsize']

    return hof


def createhistory(gen, halloffame, evaluated_individuals, opt_pipe, opt_pipe_score, gens_since_last_opt_pipe):
    """Save the current state of the process in a dict.
    (i.e. current generation, hall of fame, evaluated individuals, ...)

    :param gen: int containing the generation number
    :param halloffame: Pareto Front
    :param evaluated_individuals: Dict with already evaluated individuals

    :return history: Dict containing the state of the process.
    """

    def fmultitodict(fmulti):
        """ Write attributes of a FitnessMulti Object into a dict.

        :param fmulti: FitnessMulti Object

        :return fmultidict: Dict containing information about the FitnessMulti Object in order to save/reload it.
        """

        fmultidict = {'valid': fmulti.valid,
                      'values': fmulti.values,
                      'weights': fmulti.weights,
                      'wvalues': fmulti.wvalues}

        return fmultidict

    # keys of halloffame
    keys = []
    for key in halloffame.keys:
        fm_dict = fmultitodict(key)
        keys.append(fm_dict)

    halloffame_dict = {'items': poptodict(halloffame.items),
                       'keys': keys,
                       'maxsize': halloffame.maxsize}

    history = {'generation': gen,
               'already_evaluated': False,
               'evaluated_individuals': evaluated_individuals,
               'halloffame': halloffame_dict,
               'optimized_pipeline': opt_pipe,
               'optimized_pipeline_score': opt_pipe_score,
               'gens_since_last_opt_pipe': gens_since_last_opt_pipe}

    return history


def dicttoind(ind_dict, tpot_operators):
    """Generate an Individual by using the corresponding dict.

    Parameters
    ----------
        ind_dict: dict
            Dict containing all the information needed to reconstruct an individual.

    Returns
    -------
        tpot_ind: Individual
            Returns an Individual, which can be evaluated by TPOT.
    """

    def get_prim_attr(prim, tpot_operators):
        """Generate Attributes of a Primitive dict in order to create a <class Primitive> Object

        Parameters
        ----------
            prim: dict
                Dict containing all the information needed to reconstruct a Primitive.
            tpot_operators: list
                List of <class 'tpot.operator_utils...'> used to assign operators.

        Returns
        -------
            name: str
            args: list
            ret: tpot.Output_Array, np.ndarray or tpot.operator_utils
                Returns arguments needed to initialize a Primitive.
        """

        # assign name
        name = prim['name']

        # assign ret
        if prim['ret'] == 'Output_Array':
            ret = gp_types.Output_Array
        elif prim['ret'].__name__ == 'ndarray':
            ret = prim['ret']
        else:
            # search for matching tpot.operator_utils types
            for op in tpot_operators:
                if prim['ret'] in op.__name__:
                    ret = op

        # assign args
        args = []
        for arg in prim['args']:
            if not isinstance(arg, str):
                if arg.__name__ == 'ndarray':
                    args.append(arg)
                else:
                    pass
            else:
                # check for tpot.operator_utils types by comparing the string representation of operators in the dict
                # with tpot operators. If string matches op.__name__, add operator to args.
                for op in tpot_operators:
                    if arg == op.__name__:
                        args.append(op)
                        break  # found and added matching operator, find match for next operator string

        return name, args, ret

    def get_term_attr(term, tpot_operators):
        """Generate Attributes of a Terminal dict in order to create a <class Terminal> Object.

        Parameters
        ----------
            term: dict
                Dict containing all the information needed to generate a Terminal.
            tpot_operators: list
                List of <class 'tpot.operator_utils...'> used to assign operators.

        Returns
        -------
            terminal: str
            symbolic: bool
            ret: tpot.Output_Array, np.ndarray or tpot.operator_utils
                Returns arguments needed to initialize a Terminal.
        """

        # assign terminal
        terminal = term['name']

        # assign symbolic
        symbolic = True  # used to initialize the terminal. Will be overwritten after calling get_term_attr().

        # assign ret
        if not isinstance(term['ret'], str):
            if term['ret'].__name__ == 'ndarray':
                ret = term['ret']
            else:
                # Meines Wissens kann term['ret'] nur bei np.ndarray kein string sein.
                # Für weitere Fälle hier Code erweitern.
                pass
        else:
            if term['ret'] == 'Output_Array':
                ret = gp_types.Output_Array
            else:
                # search for matching tpot.operator_utils types
                for op in tpot_operators:
                    if term['ret'] in op.__name__:
                        ret = op

        return terminal, symbolic, ret

    gp_PrimitiveTree = []

    # loop over Primitives and Terminals of the dict representation
    # to create corresponding Primitive and Terminal Objects
    for primterm in ind_dict['prims_terms']:

        if primterm['type'] == 'Primitive':
            # generate primitive
            prim_name, prim_args, prim_ret = get_prim_attr(primterm, tpot_operators)
            primitive = gp.Primitive(prim_name, prim_args, prim_ret)
            gp_PrimitiveTree.append(primitive)
        else:
            # generate terminal
            term_terminal, term_symbolic, term_ret = get_term_attr(primterm, tpot_operators)
            terminal = gp.Terminal(term_terminal, term_symbolic, term_ret)
            terminal.conv_fct = primterm['conv_fct']
            if terminal.name == 'ARG0':
                terminal.value = 'input_matrix'
            gp_PrimitiveTree.append(terminal)

    # reconstruct individual and reassign statistics + fitness values
    tpot_ind = creator.Individual(gp_PrimitiveTree)
    tpot_ind.statistics = ind_dict['statistics']
    tpot_ind.fitness.values = ind_dict['fitness']['values']
    tpot_ind.fitness.weights = ind_dict['fitness']['weights']
    tpot_ind.fitness.wvalues = ind_dict['fitness']['wvalues']

    return tpot_ind

def poptodict(tpot_pop):
    """Prepare TPOT population for pickling by writing the attributes of each individual into a structured dict.
    The dict can later be used to reconstruct a population, in order to initialize a warmstart in case of unexpected
    crashes.

    Parameters
    ----------
        tpot_pop: list
            List of individuals, which cannot be pickled, due to TPOT classes/types.

    Returns
    -------
        pop: list
            Returns a list of individuals, which can be pickled.
    """

    pop = []
    for ind in tpot_pop:
        # create dicts for each individual
        fitness_dict = {
            "valid": ind.fitness.valid,
            "values": ind.fitness.values,
            "weights": ind.fitness.weights,
            "wvalues": ind.fitness.wvalues
        }

        # Prepare root_args for pickling:
        root_args = [arg.__name__ for arg in ind.root.args]

        root_dict = {
            "args": root_args,
            "arity": ind.root.arity,
            "name": ind.root.name,
            "ret": ind.root.ret.__name__,
            "seq": ind.root.seq,
        }

        # Collect Primitives and Terminals for each Individual in a list
        prims_terms = []
        # iterate over primitive tree (list of primitives and terminals)
        for idx in range(0, len(ind)):
            # check ret for type numpy.ndarray
            if ind[idx].ret.__name__ == "ndarray":
                ret = ind[idx].ret              # copy whole ndarray
            else:
                ret = ind[idx].ret.__name__     # other args cannot be cannot be pickled. Take names instead.

            # Prepare prim_args for pickling if type is 'Primitive'
            prim_args = []
            if ind[idx].__class__.__name__ == "Primitive":
                for arg in ind[idx].args:
                    if arg.__name__ == 'ndarray':
                        prim_args.append(arg)     # copy np.ndarray
                    else:
                        prim_args.append(arg.__name__)  # other args cannot be cannot be pickled. Take names instead.

                prim_dict = {
                    "type": "Primitive",
                    "args": prim_args,
                    "arity": ind[idx].arity,
                    "name": ind[idx].name,
                    "ret": ret,
                    "seq": ind[idx].seq
                }
                # Add Primitive
                prims_terms.append(prim_dict)

            if ind[idx].__class__.__name__ == "Terminal":
                term_dict = {
                    "type": "Terminal",
                    "arity": ind[idx].arity,
                    "conv_fct": ind[idx].conv_fct,
                    "name": ind[idx].name,
                    "ret": ret,
                    "value": ind[idx].value
                }
                # Add Terminal
                prims_terms.append(term_dict)

        # Create dict for individual and add it to the pop
        ind_dict = dict(ind=str(ind), fitness=fitness_dict, height=ind.height, root=root_dict,
                        statistics=ind.statistics, prims_terms=prims_terms)
        pop.append(ind_dict)

    return pop