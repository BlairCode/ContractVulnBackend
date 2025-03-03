import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.*;

public class Tokenize {
    public static String result;
    public String getParseString(String sourcePath) throws IOException {
        result=" ";

        InputStream is = new FileInputStream(sourcePath);
        ANTLRInputStream input = new ANTLRInputStream(is);

        SolidityLexer lexer = new SolidityLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);

        tokens.fill();

        SolidityParser parser = new SolidityParser(tokens);

        ParseTree tree = parser.sourceUnit();

        /* Extract Function Tokens */
        ParseTreeWalker walker = new ParseTreeWalker();

        // ASTSerialize listener = new ASTSerialize();
        // SolidityToSeq listener = new SolidityToSeq();
        test listener = new test();
        // ASTSerialize listener = new ASTSerialize();
        walker.walk(listener,tree);
        return result;
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("Error: No input file provided.");
            System.exit(1);
        }

        Tokenize tokenize = new Tokenize();

        try {
            // Read from file
            File file = new File(args[0]);
            if (!file.exists()) {
                System.err.println("Error: File not found -> " + args[0]);
                System.exit(1);
            }

            // Process the input
            String parsedResult = tokenize.getParseString(args[0]);
            System.out.println(parsedResult);

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return;
    }
}
