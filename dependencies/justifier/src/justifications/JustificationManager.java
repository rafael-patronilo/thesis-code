// Decompiled from Main class at https://bitbucket.org/manuelsribeiro/master_thesis/src/master/justification_platform_demo/scripts/Justifier.jar
// and modified
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package justifications;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.common.collect.HashBiMap;
import it.unife.endif.ml.bundle.Bundle;
import it.unife.endif.ml.bundle.bdd.BDDFactory2;
import it.unife.endif.ml.bundle.utilities.BundleUtilities;
import it.unife.endif.ml.math.ApproxDouble;
import it.unife.endif.ml.probowlapi.core.ProbabilisticExplanationReasonerResult;
import it.unife.endif.ml.probowlapi.exception.ObjectNotInitializedException;

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
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.InconsistentOntologyException;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.util.BidirectionalShortFormProviderAdapter;
import org.semanticweb.owlapi.util.SimpleShortFormProvider;
import org.semanticweb.owlapi.util.mansyntax.ManchesterOWLSyntaxParser;
import uk.ac.manchester.cs.owl.explanation.ordering.ExplanationOrderer;
import uk.ac.manchester.cs.owl.explanation.ordering.ExplanationOrdererImpl;
import utils.Pair;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;


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
    private List<OWLAxiom> annotatedAxioms;
    private static boolean guard = false;

    public JustificationManager(OWLOntology ontology,
                                OWLOntologyManager manager,
                                String entailment,
                                List<String> observations)
            throws InconsistentOntologyException {
        if (guard){
            throw new RuntimeException("Must clear last object first");
        }
        guard = true;
        this.ontology = ontology;
        this.manager = manager;
        this.loadEntailment(entailment);
        this.loadObservations(observations);
    }

    private void loadEntailment(String entailment) {
        Set<OWLOntology> importsClosure = this.ontology.getImportsClosure();
        OWLEntityChecker entityChecker = new ShortFormEntityChecker(new BidirectionalShortFormProviderAdapter(this.manager, importsClosure, new SimpleShortFormProvider()));
        ManchesterOWLSyntaxParser parser = OWLManager.createManchesterParser();
        parser.setDefaultOntology(this.ontology);
        parser.setOWLEntityChecker(entityChecker);
        parser.setStringToParse(entailment);
        this.entailment = parser.parseAxiom();
    }

    private void loadObservations(List<String> observations) {
        Set<OWLOntology> importsClosure = this.ontology.getImportsClosure();
        OWLEntityChecker entityChecker = new ShortFormEntityChecker(new BidirectionalShortFormProviderAdapter(this.manager, importsClosure, new SimpleShortFormProvider()));
        ManchesterOWLSyntaxParser parser = OWLManager.createManchesterParser();
        parser.setDefaultOntology(this.ontology);
        parser.setOWLEntityChecker(entityChecker);
        this.observations = new LinkedList();

        for (String observation : observations){
            String[] data = observation.split(",");
            parser.setStringToParse(data[0]);
            float confidence = Float.parseFloat(data[1]);
            this.observations.add(new Pair(parser.parseAxiom(), confidence));
        }
    }

    public void justify(int maxJustifications) throws InconsistentOntologyException {
        OWLDataFactory df = this.ontology.getOWLOntologyManager().getOWLDataFactory();
        new LinkedList();
        this.annotatedAxioms = new ArrayList<>(this.observations.size());
        this.observations.forEach((observation) -> {
            OWLAxiom obsAxiom = (OWLAxiom)observation.getKey();
            OWLAnnotation probAnnotation = df.getOWLAnnotation(df.getOWLAnnotationProperty(PROB_IRI), df.getOWLLiteral(String.valueOf(observation.getValue()), ""));
            Set<OWLAnnotation> annotations = new HashSet();
            annotations.add(probAnnotation);
            obsAxiom = obsAxiom.getAnnotatedAxiom(annotations);
            this.annotatedAxioms.add(obsAxiom);
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

    private Iterator<OWLAxiom> getOrderedAxioms(Explanation<OWLAxiom> justification){
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
        return  ((Collection)orderedAxioms).iterator();
    }

    private Set<OWLAnnotation> getBeliefAnnotations(OWLAxiom ax){
        IRI prob = IRI.create("https://sites.google.com/a/unife.it/ml/disponte#probability");
        Set<OWLAnnotation> annotations = ax.getAnnotations(this.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLAnnotationProperty(prob));
        return annotations;
    }

    private Double getBeliefValue(OWLAxiom ax){
        Set<OWLAnnotation> beliefs = getBeliefAnnotations(ax);
        boolean first = true;
        Double result = null;
        for (OWLAnnotation axiomBelief : beliefs){
            Double axiomBeliefValue = axiomBelief.getValue().asLiteral()
                    .transform(lit -> {
                        if (lit.isDouble()){
                            return lit.parseDouble();
                        } else if (lit.isFloat()) {
                            return (double)lit.parseFloat();
                        } else {
                            String toParse = lit.getLiteral();
                            return Double.parseDouble(toParse);
                        }
                    }).orNull();
            if (axiomBeliefValue != null){
                if(first){
                    result = axiomBeliefValue;
                    first = false;
                } else {
                    System.err.println("Unexpected extra belief value " + axiomBeliefValue + " is ignored");
                }
            }
        }
        return result;
    }

    public String toJSON(){
        List<Pair<Explanation<OWLAxiom>, ApproxDouble>> just = this.get_belief_degrees();
        ManchesterOWLSyntaxOWLObjectRendererImpl renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl();
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode result = mapper.createObjectNode();

        final ArrayNode loadedObservations = result.putArray("loaded_observations");
        this.observations.forEach((observation)->{
            ObjectNode obj = mapper.createObjectNode();
            obj.put("axiom", renderer.render((OWLObject)observation.getKey()));
            obj.put("belief", observation.getValue());
            loadedObservations.add(obj);
        });

        final ArrayNode justifications = result.putArray("justifications");
        just.forEach((j)->{
            Explanation<OWLAxiom> justification = (Explanation)j.getKey();
            ApproxDouble belief = (ApproxDouble)j.getValue();
            ObjectNode obj = justifications.addObject();
            obj.put("entailment", renderer.render((OWLObject)justification.getEntailment()));
            obj.put("belief", belief.getValue());
            ArrayNode usedObservations = obj.putArray("used_observations");
            ArrayNode axioms = obj.putArray("axioms");
            Iterator<OWLAxiom> orderedAxioms = getOrderedAxioms(justification);
            while(orderedAxioms.hasNext()){
                ObjectNode axiomObject = axioms.addObject();
                OWLAxiom axiom = orderedAxioms.next();
                axiomObject.put("axiom", renderer.render(axiom));
                Double beliefValue = getBeliefValue(axiom);
                if (beliefValue != null){
                    axiomObject.put("belief", beliefValue);
                }
                int i = 0;
                for(Pair<OWLAxiom, Float> observation : observations){
                    if (observation.getKey().equalsIgnoreAnnotations(axiom)){
                        usedObservations.add(i);
                        break;
                    }
                    i++;
                }
            }

        });
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        } catch (JsonProcessingException exception){
            throw new RuntimeException(exception);
        }
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
                Iterator orderedAxiomsIt = getOrderedAxioms(justification);

                while(orderedAxiomsIt.hasNext()) {
                    OWLAxiom ax = (OWLAxiom)orderedAxiomsIt.next();
                    sb.append("\t");
                    sb.append(renderer.render(ax).replaceAll("\n", "").replaceAll("\r", ""));
                    Iterator annsIt = getBeliefAnnotations(ax).iterator();

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
            bddFactory.done();
        });
        res.sort((p1, p2) -> {
            return ((ApproxDouble)p2.getValue()).compareTo((ApproxDouble)p1.getValue());
        });
        return res;
    }

    private BDD buildBDD(Set<Set<OWLAxiom>> justifications, BDDFactory bddF, Map<OWLAxiom, ApproxDouble> pMap, HashBiMap<OWLAxiom, Integer> usedAxioms) {
        return BundleUtilities.buildBDD(justifications, bddF, pMap, usedAxioms);
    }

    public void done(){
        this.annotatedAxioms.forEach(obs ->{
            this.manager.removeAxiom(this.ontology, obs);
        });
        guard = false;
    }
}
