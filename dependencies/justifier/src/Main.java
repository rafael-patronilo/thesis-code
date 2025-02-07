// Decompiled from Main class at https://bitbucket.org/manuelsribeiro/master_thesis/src/master/justification_platform_demo/scripts/Justifier.jar
// and modified
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import justifications.JustificationManager;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.InconsistentOntologyException;

public class Main {
    public Main() {
    }


    private static void justifyIteration(String entailment, List<String> observations, OWLOntologyManager man, OWLOntology o) throws IOException {
        JustificationManager jMan = new JustificationManager(o, man, entailment, observations);
        try {
            jMan.justify(Integer.MAX_VALUE);
            if (jMan.getJustifications().isEmpty()) {
                System.out.println("{\"error\" : \"not_entailed\"}");
            } else {
                String json = jMan.toJSON();
                System.out.println(json);
                System.out.println();
            }
        } catch (InconsistentOntologyException e){
            System.out.println("{\"error\" : \"inconsistent\"}");
        } finally {
            jMan.done();
        }
    }

    private static void justifierLoop(BufferedReader reader, OWLOntologyManager man, OWLOntology o)  throws IOException {
        String line = reader.readLine();
        while (line != null){
            String entailment = line;
            List<String> observations = new ArrayList<>();
            line = reader.readLine();
            while(line != null && !line.isEmpty()){
                observations.add(line);
                line = reader.readLine();
            }
            justifyIteration(entailment, observations, man, o);
            line = reader.readLine();
        }
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java -jar Justifier.jar path_to_ontology_file\n" +
            "See README.md for more info");
            System.exit(0);
        }
        BufferedReader inReader = new BufferedReader(new InputStreamReader(System.in));
        OWLOntologyManager man = OWLManager.createOWLOntologyManager();
        File file = new File(args[0]);

        try {
            OWLOntology o = man.loadOntologyFromOntologyDocument(file);
            justifierLoop(inReader, man, o);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
