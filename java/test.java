
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ParseTree;

import java.util.List;

public class test extends SolidityBaseListener{
//    这里的ctx就是每个含有孩子的节点对象,简单的;诸如此类的也是被认为是org.antlr.v4.runtime.tree.TerminalNodeImpl
    public void enterEveryRule(ParserRuleContext ctx){
        String parentNode = ctx.getClass().getName().split("\\$")[1];
        String temp = adjustName(parentNode) + " ";
        Tokenize.result += temp;
        List<ParseTree> children = ctx.children;
//        System.out.println(temp+"----->"+children.size());
        if(children != null){
            for (ParseTree child : children) {
                String name = child.getClass().getName();
                if(name.equals("org.antlr.v4.runtime.tree.TerminalNodeImpl")){
                    temp = child.toString();
                    if(temp.equals("<EOF>")){
                        temp = "";
                    }else {
                        temp = " "+temp;
                    }
                    Tokenize.result= Tokenize.result + temp + " ";
                }else{
                    temp = adjustName(name.split("\\$")[1]);
                    Tokenize.result= Tokenize.result + temp + " ";
                }
            }
            Tokenize.result = Tokenize.result + " ";
        }
        Tokenize.result = Tokenize.result +" ";
    }

    public String adjustName(String s) {
        return s.substring(0, s.length() - 7);
    }

}
