# Justifier

Command line tool using the bundle reasoner. 

This is a modified version of the tool at https://bitbucket.org/manuelsribeiro/master_thesis/src/master/justification_platform_demo/scripts/Justifier.jar

# How to build

`cd ..; ./docker-build` will build a docker image which includes this command line tool compiled with the required dependencies.


# How to use

Usage `java -jar Justifier.jar ontology_file`

To justify an entailment given observations, the tool input should be:
```
EntailmentAxiom\n
ObservationAxiom, Belief\n
ObservationAxiom, Belief\n
...
ObservationAxiom, Belief\n
\n
```

where axioms are specified in manchester syntax. For example:
```
__input__ Type: (TypeA)
__input__ Type: (has some PassengerCar), 0.6
__input__ Type: (has some ReinforcedCar), 0.8

```
will justify the entailment `__input__ Type: (TypeA)` given the observations.

Output will be a json object with the following strucutre:

```json
{
  "loaded_observations" : [
    {
      "axiom" : "...",
      "belief" : ...,
    },
    ...
  ],
  "justifications" : [
    {
      "entailment" : "...",
      "belief" : ...,
      "used_observations" : [
        ...
      ],
      "axioms" : [
        {
          "axiom": "...",
          "belief": [...]
        },
        ...
      ]
    },
    ...
  ]
}

```

To facilitate interaction with other programs, multiple queries can be made. 
Different query inputs/outputs are separated by an empty line.
The tool answers each query live, so the output will be printed as soon as the query is processed.

# Tests

The project uses no test framework but has simple bash input/output tests that are available in the directory tests.
They can be run from the repository root with `./dependencies/justifier/tests/tests.sh`
