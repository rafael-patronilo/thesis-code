//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import justifications.JustificationManager;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.InconsistentOntologyException;

public class Main {
    public Main() {
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Proper Usage: java program path_to_ontology_file path_to_observations_file path_to_output_file");
            System.exit(0);
        }

        OWLOntologyManager man = OWLManager.createOWLOntologyManager();
        File file = new File(args[0]);

        try {
            OWLOntology o = man.loadOntologyFromOntologyDocument(file);
            JustificationManager jMan = new JustificationManager(o, man, args[1], Integer.MAX_VALUE);
            String justification = jMan.toPrettyString();

            try {
                PrintWriter out = new PrintWriter(args[2]);
                out.println(justification);
                out.close();
            } catch (FileNotFoundException var7) {
                FileNotFoundException e = var7;
                e.printStackTrace();
            }
        } catch (OWLOntologyCreationException var8) {
            OWLOntologyCreationException e = var8;
            e.printStackTrace();
        } catch (InconsistentOntologyException var9) {
            System.out.println("J:I");
        }

    }
}
