//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package justifications;

import com.google.common.collect.HashBiMap;
import it.unife.endif.ml.bundle.Bundle;
import it.unife.endif.ml.bundle.bdd.BDDFactory2;
import it.unife.endif.ml.bundle.utilities.BundleUtilities;
import it.unife.endif.ml.math.ApproxDouble;
import it.unife.endif.ml.probowlapi.core.ProbabilisticExplanationReasonerResult;
import it.unife.endif.ml.probowlapi.exception.ObjectNotInitializedException;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import net.sf.javabdd.BDD;
import net.sf.javabdd.BDDFactory;
import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owl.explanation.api.Explanation;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.expression.OWLEntityChecker;
import org.semanticweb.owlapi.expression.ShortFormEntityChecker;
import org.semanticweb.owlapi.manchestersyntax.renderer.ManchesterOWLSyntaxOWLObjectRendererImpl;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLAnnotation;
import org.semanticweb.owlapi.model.OWLAxiom;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLException;
import org.semanticweb.owlapi.model.OWLObject;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.InconsistentOntologyException;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.util.BidirectionalShortFormProviderAdapter;
import org.semanticweb.owlapi.util.SimpleShortFormProvider;
import org.semanticweb.owlapi.util.mansyntax.ManchesterOWLSyntaxParser;
import uk.ac.manchester.cs.owl.explanation.ordering.ExplanationOrderer;
import uk.ac.manchester.cs.owl.explanation.ordering.ExplanationOrdererImpl;
import utils.Pair;

public class JustificationManager {
    private static final IRI PROB_IRI = IRI.create("https://sites.google.com/a/unife.it/ml/disponte#probability");
    private static final String BDD_PACKAGE = "buddy";
    private static final int NODE_NUM = 100;
    private static final int CACHE_SIZE = 10000;
    private OWLAxiom entailment;
    private List<Pair<OWLAxiom, Float>> observations;
    private OWLOntology ontology;
    private OWLOntologyManager manager;
    private List<Explanation<OWLAxiom>> justifications;

    public JustificationManager(OWLOntology ontology, OWLOntologyManager manager, String path_to_observations, int maxJustifications) throws InconsistentOntologyException {
        this.ontology = ontology;
        this.manager = manager;
        this.loadEntailment(path_to_observations);
        this.loadObservations(path_to_observations);
        this.justify(maxJustifications);
    }

    private void loadEntailment(String path_to_observations) {
        Set<OWLOntology> importsClosure = this.ontology.getImportsClosure();
        OWLEntityChecker entityChecker = new ShortFormEntityChecker(new BidirectionalShortFormProviderAdapter(this.manager, importsClosure, new SimpleShortFormProvider()));
        ManchesterOWLSyntaxParser parser = OWLManager.createManchesterParser();
        parser.setDefaultOntology(this.ontology);
        parser.setOWLEntityChecker(entityChecker);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path_to_observations));
            parser.setStringToParse(reader.readLine());
            this.entailment = parser.parseAxiom();
            reader.close();
        } catch (FileNotFoundException var6) {
            FileNotFoundException e = var6;
            e.printStackTrace();
            System.out.println("[Error] Observations file not found.");
        } catch (IOException var7) {
            IOException e = var7;
            e.printStackTrace();
            System.out.println("[Error] I/O Error reading observed entailment from observations file.");
        }

    }

    private void loadObservations(String path_to_observations) {
        Set<OWLOntology> importsClosure = this.ontology.getImportsClosure();
        OWLEntityChecker entityChecker = new ShortFormEntityChecker(new BidirectionalShortFormProviderAdapter(this.manager, importsClosure, new SimpleShortFormProvider()));
        ManchesterOWLSyntaxParser parser = OWLManager.createManchesterParser();
        parser.setDefaultOntology(this.ontology);
        parser.setOWLEntityChecker(entityChecker);
        this.observations = new LinkedList();

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path_to_observations));
            reader.readLine();

            String row;
            while((row = reader.readLine()) != null) {
                String[] data = row.split(",");
                parser.setStringToParse(data[0]);
                float confidence = Float.parseFloat(data[1]);
                this.observations.add(new Pair(parser.parseAxiom(), confidence));
            }

            reader.close();
        } catch (FileNotFoundException var9) {
            FileNotFoundException e = var9;
            e.printStackTrace();
            System.out.println("[Error] Observations file not found.");
        } catch (IOException var10) {
            IOException e = var10;
            e.printStackTrace();
            System.out.println("[Error] I/O Error reading observations from observations file.");
        }

    }

    private void justify(int maxJustifications) throws InconsistentOntologyException {
        OWLDataFactory df = this.ontology.getOWLOntologyManager().getOWLDataFactory();
        new LinkedList();
        this.observations.forEach((observation) -> {
            OWLAxiom obsAxiom = (OWLAxiom)observation.getKey();
            OWLAnnotation probAnnotation = df.getOWLAnnotation(df.getOWLAnnotationProperty(PROB_IRI), df.getOWLLiteral(String.valueOf(observation.getValue()), ""));
            Set<OWLAnnotation> annotations = new HashSet();
            annotations.add(probAnnotation);
            obsAxiom = obsAxiom.getAnnotatedAxiom(annotations);
            this.manager.addAxiom(this.ontology, obsAxiom);
        });
        OWLReasonerFactory rf = new ReasonerFactory();
        Bundle reasoner = new Bundle();
        reasoner.setReasonerFactory(rf);
        reasoner.setRootOntology(this.ontology);
        reasoner.setMaxExplanations(maxJustifications);
        reasoner.init();
        ProbabilisticExplanationReasonerResult result = null;

        try {
            result = reasoner.computeExplainQuery(this.entailment);
        } catch (OWLException var8) {
            OWLException e = var8;
            e.printStackTrace();
        } catch (ObjectNotInitializedException var9) {
            ObjectNotInitializedException e = var9;
            e.printStackTrace();
        }

        reasoner.dispose();
        this.justifications = new ArrayList();
        result.getQueryExplanations().forEach((just) -> {
            this.justifications.add(new Explanation(this.entailment, just));
        });
    }

    public OWLAxiom getEntailment() {
        return this.entailment;
    }

    public List<Pair<OWLAxiom, Float>> getObservations() {
        return this.observations;
    }

    public List<Explanation<OWLAxiom>> getJustifications() {
        return this.justifications;
    }

    public String getJustificationConceptList() {
        List<Pair<Explanation<OWLAxiom>, ApproxDouble>> just = this.get_belief_degrees();
        ManchesterOWLSyntaxOWLObjectRendererImpl renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        StringBuilder sb = new StringBuilder();
        just.forEach((j) -> {
            sb.append("J:");
            Explanation<OWLAxiom> justification = (Explanation)j.getKey();
            ApproxDouble belief = (ApproxDouble)j.getValue();
            if (justification.isEmpty()) {
                sb.append("<Empty>\n");
            } else {
                Object orderedAxioms;
                if (justification.getEntailment() instanceof OWLAxiom) {
                    OWLAxiom entailedAxiom = (OWLAxiom)justification.getEntailment();
                    ExplanationOrderer orderer = new ExplanationOrdererImpl(OWLManager.createOWLOntologyManager());
                    List<OWLAxiom> axs = new ArrayList(orderer.getOrderedExplanation(entailedAxiom, justification.getAxioms()).fillDepthFirst());
                    axs.remove(0);
                    orderedAxioms = axs;
                } else {
                    orderedAxioms = new TreeSet(justification.getAxioms());
                }

                Iterator orderedAxiomsIt = ((Collection)orderedAxioms).iterator();

                while(orderedAxiomsIt.hasNext()) {
                    OWLAxiom ax = (OWLAxiom)orderedAxiomsIt.next();
                    IRI prob = IRI.create("https://sites.google.com/a/unife.it/ml/disponte#probability");
                    Set<OWLAnnotation> annotations = ax.getAnnotations(this.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLAnnotationProperty(prob));
                    if (!annotations.isEmpty()) {
                        sb.append(renderer.render(ax).replaceAll("__input__ Type |\\n|\\r|\\r\\n", ""));
                        sb.append(";");
                    }
                }
            }

        });
        return sb.toString();
    }

    public String toPrettyString() {
        List<Pair<Explanation<OWLAxiom>, ApproxDouble>> just = this.get_belief_degrees();
        ManchesterOWLSyntaxOWLObjectRendererImpl renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        StringBuilder sb = new StringBuilder();
        sb.append("Read Observations: \n");
        this.observations.forEach((observation) -> {
            sb.append("\t");
            sb.append(renderer.render((OWLObject)observation.getKey()));
            sb.append("\n");
        });
        sb.append("\n");
        sb.append("Found Justifications:\n");
        just.forEach((j) -> {
            Explanation<OWLAxiom> justification = (Explanation)j.getKey();
            ApproxDouble belief = (ApproxDouble)j.getValue();
            sb.append("Justification for '");
            sb.append(renderer.render((OWLObject)justification.getEntailment()));
            sb.append("':\t");
            sb.append("(Degree of Belief: ");
            sb.append(belief);
            sb.append(")\n");
            if (justification.isEmpty()) {
                sb.append("\t<Empty>\n");
            } else {
                Object orderedAxioms;
                if (justification.getEntailment() instanceof OWLAxiom) {
                    OWLAxiom entailedAxiom = (OWLAxiom)justification.getEntailment();
                    ExplanationOrderer orderer = new ExplanationOrdererImpl(OWLManager.createOWLOntologyManager());
                    List<OWLAxiom> axs = new ArrayList(orderer.getOrderedExplanation(entailedAxiom, justification.getAxioms()).fillDepthFirst());
                    axs.remove(0);
                    orderedAxioms = axs;
                } else {
                    orderedAxioms = new TreeSet(justification.getAxioms());
                }

                Iterator orderedAxiomsIt = ((Collection)orderedAxioms).iterator();

                while(orderedAxiomsIt.hasNext()) {
                    OWLAxiom ax = (OWLAxiom)orderedAxiomsIt.next();
                    IRI prob = IRI.create("https://sites.google.com/a/unife.it/ml/disponte#probability");
                    Set<OWLAnnotation> annotations = ax.getAnnotations(this.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLAnnotationProperty(prob));
                    sb.append("\t");
                    sb.append(renderer.render(ax).replaceAll("\n", "").replaceAll("\r", ""));
                    Iterator annsIt = annotations.iterator();

                    while(annsIt.hasNext()) {
                        OWLAnnotation annotation = (OWLAnnotation)annsIt.next();
                        sb.append("\t");
                        sb.append("(" + annotation.getValue() + ")");
                    }

                    sb.append("\n");
                }
            }

            sb.append("\n");
        });
        return sb.toString();
    }

    public List<Pair<Explanation<OWLAxiom>, ApproxDouble>> get_belief_degrees() {
        List<Pair<Explanation<OWLAxiom>, ApproxDouble>> res = new ArrayList();
        this.justifications.forEach((just) -> {
            BDDFactory bddFactory = BDDFactory2.init("buddy", 100, 10000);
            Map<OWLAxiom, ApproxDouble> pMap = BundleUtilities.createPMap(this.ontology, false);
            HashBiMap<OWLAxiom, Integer> usedAxioms = HashBiMap.create();
            BDD bdd = this.buildBDD(Collections.singleton(just.getAxioms()), bddFactory, pMap, usedAxioms);
            ApproxDouble prob = BundleUtilities.probabilityOfBDD(bdd, new HashMap(), pMap, usedAxioms);
            res.add(new Pair(just, prob));
        });
        res.sort((p1, p2) -> {
            return ((ApproxDouble)p2.getValue()).compareTo((ApproxDouble)p1.getValue());
        });
        return res;
    }

    private BDD buildBDD(Set<Set<OWLAxiom>> justifications, BDDFactory bddF, Map<OWLAxiom, ApproxDouble> pMap, HashBiMap<OWLAxiom, Integer> usedAxioms) {
        return BundleUtilities.buildBDD(justifications, bddF, pMap, usedAxioms);
    }
}
