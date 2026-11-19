// Project-owned observer of the official Pilot Implementation's EMF model.
// The Pilot Implementation remains responsible for SysML parsing and resolution.
import java.nio.file.*;
import java.util.*;
import org.eclipse.emf.ecore.*;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.xtext.nodemodel.util.NodeModelUtils;
import org.eclipse.xtext.resource.IResourceServiceProvider;
import org.eclipse.xtext.validation.*;
import org.eclipse.xtext.util.CancelIndicator;
import org.omg.sysml.lang.sysml.Element;
import org.omg.sysml.xtext.util.SysML2JSON;
import org.omg.sysml.util.SysMLLibraryUtil;
import com.google.gson.*;

public class CompilerExport {
  static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
  static Path input;
  static final Set<String> DERIVED = Set.of("name", "qualifiedName", "owner", "type",
      "value", "ownedFeature", "isComposite", "isReference", "satisfiedRequirement",
      "satisfyingFeature", "subjectParameter", "referent");

  static Map<String,Object> identity(Element e) {
    Map<String,Object> m = new LinkedHashMap<>();
    m.put("id", e.getElementId()); m.put("class", e.eClass().getName());
    m.put("declaredName", e.getDeclaredName()); m.put("proxy", e.eIsProxy());
    if(e.eResource()!=null) {
      Resource r=e.eResource();
      Path file=Path.of(r.getURI().toFileString()).toAbsolutePath().normalize();
      boolean local=file.startsWith(input);
      m.put("resource", file.toString()); m.put("input", local);
      m.put("source_key", (local ? "input:"+input.relativize(file).toString().replace('\\','/') : "library:"+file.toString().replace('\\','/'))
          +"#"+r.getURIFragment(e));
    }
    return m;
  }
  static Object value(Object v) {
    if(v instanceof Element e)return identity(e);
    if(v instanceof Collection<?> c)return c.stream().map(CompilerExport::value).toList();
    if(v instanceof EObject e)return e.toString();
    return v;
  }
  static void load(SysML2JSON tool, Path root, boolean isInput) throws Exception {
    try(var paths=Files.walk(root)) {
      for(Path p:paths.filter(Files::isRegularFile).filter(p->p.toString().endsWith(".sysml") || p.toString().endsWith(".kerml")).sorted().toList()) {
        Resource r=tool.getResource(p.toAbsolutePath().normalize().toString());
        if(isInput)tool.addInputResource(r);
      }
    }
  }
  public static void main(String[] args) throws Exception {
    // input directory, output JSON file, standard library directory, extra library directories
    if(args.length<3)throw new IllegalArgumentException("input output standard-library [library ...]");
    input=Path.of(args[0]).toAbsolutePath().normalize();
    SysMLLibraryUtil.setModelLibraryDirectory(Path.of(args[2]).toAbsolutePath().toString());
    SysML2JSON tool=new SysML2JSON();
    for(int i=2;i<args.length;i++)load(tool,Path.of(args[i]),false);
    load(tool,input,true);
    System.out.println("Loaded "+tool.getInputResources().size()+" input files; transforming");
    tool.transformAll(false);
    List<Object> issues=new ArrayList<>();
    for(Resource r:tool.getInputResources()) {
      System.out.println("Validating "+r.getURI().lastSegment());
      var validator=IResourceServiceProvider.Registry.INSTANCE.getResourceServiceProvider(r.getURI()).getResourceValidator();
      for(Issue i:validator.validate(r,CheckMode.ALL,CancelIndicator.NullImpl)) {
        Map<String,Object> issue=new LinkedHashMap<>();
        issue.put("file",input.relativize(Path.of(r.getURI().toFileString()).toAbsolutePath().normalize()).toString().replace('\\','/'));
        issue.put("severity",i.getSeverity().toString());issue.put("line",i.getLineNumber());
        issue.put("code",i.getCode());issue.put("message",i.getMessage());issues.add(issue);
      }
    }
    List<Object> rows=new ArrayList<>(), propertyErrors=new ArrayList<>();
    for(Resource r:tool.getInputResources()) {
      // Resource.getAllContents includes roots: do not duplicate them.
      var objects=r.getAllContents();
      while(objects.hasNext())if(objects.next() instanceof Element e) {
        Map<String,Object> row=identity(e);
        row.put("file",input.relativize(Path.of(r.getURI().toFileString()).toAbsolutePath().normalize()).toString().replace('\\','/'));
        var node=NodeModelUtils.findActualNodeFor(e);
        if(node!=null){row.put("line",node.getStartLine());row.put("end_line",node.getEndLine());row.put("source",node.getText());row.put("offset",node.getOffset());}
        row.put("source_backed",node!=null);
        Map<String,Object> fields=new LinkedHashMap<>(), origins=new LinkedHashMap<>();
        for(EStructuralFeature f:e.eClass().getEAllStructuralFeatures())if(!f.isDerived() || DERIVED.contains(f.getName())) {
          try {
            Object v=e.eGet(f);
            if(v!=null && (!(v instanceof Collection<?>) || !((Collection<?>)v).isEmpty())) {
              fields.put(f.getName(),value(v));origins.put(f.getName(),f.isDerived()?"computed":"stored");
            }
          }catch(RuntimeException ex){propertyErrors.add(Map.of("id",e.getElementId(),"property",f.getName(),"message",ex.toString()));}
        }
        row.put("fields",fields);row.put("field_origins",origins);rows.add(row);
      }
    }
    List<Object> libraryErrors=new ArrayList<>();
    for(Resource r:tool.getLibraryResources())for(var d:r.getErrors())
      libraryErrors.add(Map.of("resource",r.getURI().toString(),"line",d.getLine(),"message",d.getMessage()));
    Map<String,Object> result=new LinkedHashMap<>();
    result.put("schema","sysml-compiler-observer-v1");result.put("elements",rows);
    result.put("validation",issues);result.put("property_errors",propertyErrors);result.put("library_resource_errors",libraryErrors);
    result.put("implicit_generation_requested",false);
    Files.writeString(Path.of(args[1]),GSON.toJson(result));
    System.out.println("Exported "+rows.size()+" semantic objects and "+issues.size()+" input issues");
  }
}
