// Generated from ../java/Solidity.g4 by ANTLR 4.13.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue", "this-escape"})
public class SolidityParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, T__21=22, T__22=23, T__23=24, 
		T__24=25, T__25=26, T__26=27, T__27=28, T__28=29, T__29=30, T__30=31, 
		T__31=32, T__32=33, T__33=34, T__34=35, T__35=36, T__36=37, T__37=38, 
		T__38=39, T__39=40, T__40=41, T__41=42, T__42=43, T__43=44, T__44=45, 
		T__45=46, T__46=47, T__47=48, T__48=49, T__49=50, T__50=51, T__51=52, 
		T__52=53, T__53=54, T__54=55, T__55=56, T__56=57, T__57=58, T__58=59, 
		T__59=60, T__60=61, T__61=62, T__62=63, T__63=64, T__64=65, T__65=66, 
		T__66=67, T__67=68, T__68=69, T__69=70, T__70=71, T__71=72, T__72=73, 
		T__73=74, T__74=75, T__75=76, T__76=77, T__77=78, T__78=79, T__79=80, 
		T__80=81, T__81=82, T__82=83, T__83=84, T__84=85, T__85=86, T__86=87, 
		T__87=88, T__88=89, T__89=90, T__90=91, T__91=92, T__92=93, T__93=94, 
		T__94=95, T__95=96, T__96=97, Int=98, Uint=99, Byte=100, Fixed=101, Ufixed=102, 
		BooleanLiteral=103, DecimalNumber=104, HexNumber=105, NumberUnit=106, 
		HexLiteralFragment=107, ReservedKeyword=108, AnonymousKeyword=109, BreakKeyword=110, 
		ConstantKeyword=111, ImmutableKeyword=112, ContinueKeyword=113, LeaveKeyword=114, 
		ExternalKeyword=115, IndexedKeyword=116, InternalKeyword=117, PayableKeyword=118, 
		PrivateKeyword=119, PublicKeyword=120, VirtualKeyword=121, PureKeyword=122, 
		TypeKeyword=123, ViewKeyword=124, ConstructorKeyword=125, FallbackKeyword=126, 
		ReceiveKeyword=127, Identifier=128, StringLiteralFragment=129, VersionLiteral=130, 
		WS=131, COMMENT=132, LINE_COMMENT=133;
	public static final int
		RULE_sourceUnit = 0, RULE_pragmaDirective = 1, RULE_pragmaName = 2, RULE_pragmaValue = 3, 
		RULE_version = 4, RULE_versionOperator = 5, RULE_versionConstraint = 6, 
		RULE_importDeclaration = 7, RULE_importDirective = 8, RULE_importPath = 9, 
		RULE_contractDefinition = 10, RULE_inheritanceSpecifier = 11, RULE_contractPart = 12, 
		RULE_stateVariableDeclaration = 13, RULE_fileLevelConstant = 14, RULE_customErrorDefinition = 15, 
		RULE_usingForDeclaration = 16, RULE_structDefinition = 17, RULE_modifierDefinition = 18, 
		RULE_modifierInvocation = 19, RULE_functionDefinition = 20, RULE_functionDescriptor = 21, 
		RULE_returnParameters = 22, RULE_modifierList = 23, RULE_eventDefinition = 24, 
		RULE_enumValue = 25, RULE_enumDefinition = 26, RULE_parameterList = 27, 
		RULE_parameter = 28, RULE_eventParameterList = 29, RULE_eventParameter = 30, 
		RULE_functionTypeParameterList = 31, RULE_functionTypeParameter = 32, 
		RULE_variableDeclaration = 33, RULE_typeName = 34, RULE_userDefinedTypeName = 35, 
		RULE_mappingKey = 36, RULE_mapping = 37, RULE_functionTypeName = 38, RULE_storageLocation = 39, 
		RULE_stateMutability = 40, RULE_block = 41, RULE_statement = 42, RULE_expressionStatement = 43, 
		RULE_ifStatement = 44, RULE_tryStatement = 45, RULE_catchClause = 46, 
		RULE_whileStatement = 47, RULE_simpleStatement = 48, RULE_uncheckedStatement = 49, 
		RULE_forStatement = 50, RULE_inlineAssemblyStatement = 51, RULE_doWhileStatement = 52, 
		RULE_continueStatement = 53, RULE_breakStatement = 54, RULE_returnStatement = 55, 
		RULE_throwStatement = 56, RULE_emitStatement = 57, RULE_revertStatement = 58, 
		RULE_variableDeclarationStatement = 59, RULE_variableDeclarationList = 60, 
		RULE_identifierList = 61, RULE_elementaryTypeName = 62, RULE_expression = 63, 
		RULE_primaryExpression = 64, RULE_expressionList = 65, RULE_nameValueList = 66, 
		RULE_nameValue = 67, RULE_functionCallArguments = 68, RULE_functionCall = 69, 
		RULE_assemblyBlock = 70, RULE_assemblyItem = 71, RULE_assemblyExpression = 72, 
		RULE_assemblyMember = 73, RULE_assemblyCall = 74, RULE_assemblyLocalDefinition = 75, 
		RULE_assemblyAssignment = 76, RULE_assemblyIdentifierOrList = 77, RULE_assemblyIdentifierList = 78, 
		RULE_assemblyStackAssignment = 79, RULE_labelDefinition = 80, RULE_assemblySwitch = 81, 
		RULE_assemblyCase = 82, RULE_assemblyFunctionDefinition = 83, RULE_assemblyFunctionReturns = 84, 
		RULE_assemblyFor = 85, RULE_assemblyIf = 86, RULE_assemblyLiteral = 87, 
		RULE_subAssembly = 88, RULE_tupleExpression = 89, RULE_typeNameExpression = 90, 
		RULE_numberLiteral = 91, RULE_identifier = 92, RULE_hexLiteral = 93, RULE_overrideSpecifier = 94, 
		RULE_stringLiteral = 95;
	private static String[] makeRuleNames() {
		return new String[] {
			"sourceUnit", "pragmaDirective", "pragmaName", "pragmaValue", "version", 
			"versionOperator", "versionConstraint", "importDeclaration", "importDirective", 
			"importPath", "contractDefinition", "inheritanceSpecifier", "contractPart", 
			"stateVariableDeclaration", "fileLevelConstant", "customErrorDefinition", 
			"usingForDeclaration", "structDefinition", "modifierDefinition", "modifierInvocation", 
			"functionDefinition", "functionDescriptor", "returnParameters", "modifierList", 
			"eventDefinition", "enumValue", "enumDefinition", "parameterList", "parameter", 
			"eventParameterList", "eventParameter", "functionTypeParameterList", 
			"functionTypeParameter", "variableDeclaration", "typeName", "userDefinedTypeName", 
			"mappingKey", "mapping", "functionTypeName", "storageLocation", "stateMutability", 
			"block", "statement", "expressionStatement", "ifStatement", "tryStatement", 
			"catchClause", "whileStatement", "simpleStatement", "uncheckedStatement", 
			"forStatement", "inlineAssemblyStatement", "doWhileStatement", "continueStatement", 
			"breakStatement", "returnStatement", "throwStatement", "emitStatement", 
			"revertStatement", "variableDeclarationStatement", "variableDeclarationList", 
			"identifierList", "elementaryTypeName", "expression", "primaryExpression", 
			"expressionList", "nameValueList", "nameValue", "functionCallArguments", 
			"functionCall", "assemblyBlock", "assemblyItem", "assemblyExpression", 
			"assemblyMember", "assemblyCall", "assemblyLocalDefinition", "assemblyAssignment", 
			"assemblyIdentifierOrList", "assemblyIdentifierList", "assemblyStackAssignment", 
			"labelDefinition", "assemblySwitch", "assemblyCase", "assemblyFunctionDefinition", 
			"assemblyFunctionReturns", "assemblyFor", "assemblyIf", "assemblyLiteral", 
			"subAssembly", "tupleExpression", "typeNameExpression", "numberLiteral", 
			"identifier", "hexLiteral", "overrideSpecifier", "stringLiteral"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'pragma'", "';'", "'||'", "'^'", "'~'", "'>='", "'>'", "'<'", 
			"'<='", "'='", "'as'", "'import'", "'*'", "'from'", "'{'", "','", "'}'", 
			"'abstract'", "'contract'", "'interface'", "'library'", "'is'", "'('", 
			"')'", "'error'", "'using'", "'for'", "'struct'", "'modifier'", "'function'", 
			"'returns'", "'event'", "'enum'", "'['", "']'", "'address'", "'.'", "'mapping'", 
			"'=>'", "'memory'", "'storage'", "'calldata'", "'if'", "'else'", "'try'", 
			"'catch'", "'while'", "'unchecked'", "'assembly'", "'do'", "'return'", 
			"'throw'", "'emit'", "'revert'", "'var'", "'bool'", "'string'", "'byte'", 
			"'++'", "'--'", "'new'", "':'", "'+'", "'-'", "'after'", "'delete'", 
			"'!'", "'**'", "'/'", "'%'", "'<<'", "'>>'", "'&'", "'|'", "'=='", "'!='", 
			"'&&'", "'?'", "'|='", "'^='", "'&='", "'<<='", "'>>='", "'+='", "'-='", 
			"'*='", "'/='", "'%='", "'let'", "':='", "'=:'", "'switch'", "'case'", 
			"'default'", "'->'", "'callback'", "'override'", null, null, null, null, 
			null, null, null, null, null, null, null, "'anonymous'", "'break'", "'constant'", 
			"'immutable'", "'continue'", "'leave'", "'external'", "'indexed'", "'internal'", 
			"'payable'", "'private'", "'public'", "'virtual'", "'pure'", "'type'", 
			"'view'", "'constructor'", "'fallback'", "'receive'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, "Int", "Uint", "Byte", "Fixed", "Ufixed", "BooleanLiteral", 
			"DecimalNumber", "HexNumber", "NumberUnit", "HexLiteralFragment", "ReservedKeyword", 
			"AnonymousKeyword", "BreakKeyword", "ConstantKeyword", "ImmutableKeyword", 
			"ContinueKeyword", "LeaveKeyword", "ExternalKeyword", "IndexedKeyword", 
			"InternalKeyword", "PayableKeyword", "PrivateKeyword", "PublicKeyword", 
			"VirtualKeyword", "PureKeyword", "TypeKeyword", "ViewKeyword", "ConstructorKeyword", 
			"FallbackKeyword", "ReceiveKeyword", "Identifier", "StringLiteralFragment", 
			"VersionLiteral", "WS", "COMMENT", "LINE_COMMENT"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "Solidity.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public SolidityParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class SourceUnitContext extends ParserRuleContext {
		public TerminalNode EOF() { return getToken(SolidityParser.EOF, 0); }
		public List<PragmaDirectiveContext> pragmaDirective() {
			return getRuleContexts(PragmaDirectiveContext.class);
		}
		public PragmaDirectiveContext pragmaDirective(int i) {
			return getRuleContext(PragmaDirectiveContext.class,i);
		}
		public List<ImportDirectiveContext> importDirective() {
			return getRuleContexts(ImportDirectiveContext.class);
		}
		public ImportDirectiveContext importDirective(int i) {
			return getRuleContext(ImportDirectiveContext.class,i);
		}
		public List<ContractDefinitionContext> contractDefinition() {
			return getRuleContexts(ContractDefinitionContext.class);
		}
		public ContractDefinitionContext contractDefinition(int i) {
			return getRuleContext(ContractDefinitionContext.class,i);
		}
		public List<EnumDefinitionContext> enumDefinition() {
			return getRuleContexts(EnumDefinitionContext.class);
		}
		public EnumDefinitionContext enumDefinition(int i) {
			return getRuleContext(EnumDefinitionContext.class,i);
		}
		public List<StructDefinitionContext> structDefinition() {
			return getRuleContexts(StructDefinitionContext.class);
		}
		public StructDefinitionContext structDefinition(int i) {
			return getRuleContext(StructDefinitionContext.class,i);
		}
		public List<FunctionDefinitionContext> functionDefinition() {
			return getRuleContexts(FunctionDefinitionContext.class);
		}
		public FunctionDefinitionContext functionDefinition(int i) {
			return getRuleContext(FunctionDefinitionContext.class,i);
		}
		public List<FileLevelConstantContext> fileLevelConstant() {
			return getRuleContexts(FileLevelConstantContext.class);
		}
		public FileLevelConstantContext fileLevelConstant(int i) {
			return getRuleContext(FileLevelConstantContext.class,i);
		}
		public List<CustomErrorDefinitionContext> customErrorDefinition() {
			return getRuleContexts(CustomErrorDefinitionContext.class);
		}
		public CustomErrorDefinitionContext customErrorDefinition(int i) {
			return getRuleContext(CustomErrorDefinitionContext.class,i);
		}
		public SourceUnitContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_sourceUnit; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterSourceUnit(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitSourceUnit(this);
		}
	}

	public final SourceUnitContext sourceUnit() throws RecognitionException {
		SourceUnitContext _localctx = new SourceUnitContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_sourceUnit);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(202);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451105407455234L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 8057520253L) != 0)) {
				{
				setState(200);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,0,_ctx) ) {
				case 1:
					{
					setState(192);
					pragmaDirective();
					}
					break;
				case 2:
					{
					setState(193);
					importDirective();
					}
					break;
				case 3:
					{
					setState(194);
					contractDefinition();
					}
					break;
				case 4:
					{
					setState(195);
					enumDefinition();
					}
					break;
				case 5:
					{
					setState(196);
					structDefinition();
					}
					break;
				case 6:
					{
					setState(197);
					functionDefinition();
					}
					break;
				case 7:
					{
					setState(198);
					fileLevelConstant();
					}
					break;
				case 8:
					{
					setState(199);
					customErrorDefinition();
					}
					break;
				}
				}
				setState(204);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(205);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PragmaDirectiveContext extends ParserRuleContext {
		public PragmaNameContext pragmaName() {
			return getRuleContext(PragmaNameContext.class,0);
		}
		public PragmaValueContext pragmaValue() {
			return getRuleContext(PragmaValueContext.class,0);
		}
		public PragmaDirectiveContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pragmaDirective; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterPragmaDirective(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitPragmaDirective(this);
		}
	}

	public final PragmaDirectiveContext pragmaDirective() throws RecognitionException {
		PragmaDirectiveContext _localctx = new PragmaDirectiveContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_pragmaDirective);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(207);
			match(T__0);
			setState(208);
			pragmaName();
			setState(209);
			pragmaValue();
			setState(210);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PragmaNameContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public PragmaNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pragmaName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterPragmaName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitPragmaName(this);
		}
	}

	public final PragmaNameContext pragmaName() throws RecognitionException {
		PragmaNameContext _localctx = new PragmaNameContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_pragmaName);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(212);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PragmaValueContext extends ParserRuleContext {
		public VersionContext version() {
			return getRuleContext(VersionContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public PragmaValueContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pragmaValue; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterPragmaValue(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitPragmaValue(this);
		}
	}

	public final PragmaValueContext pragmaValue() throws RecognitionException {
		PragmaValueContext _localctx = new PragmaValueContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_pragmaValue);
		try {
			setState(216);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(214);
				version();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(215);
				expression(0);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VersionContext extends ParserRuleContext {
		public List<VersionConstraintContext> versionConstraint() {
			return getRuleContexts(VersionConstraintContext.class);
		}
		public VersionConstraintContext versionConstraint(int i) {
			return getRuleContext(VersionConstraintContext.class,i);
		}
		public VersionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_version; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVersion(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVersion(this);
		}
	}

	public final VersionContext version() throws RecognitionException {
		VersionContext _localctx = new VersionContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_version);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(218);
			versionConstraint();
			setState(225);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2040L) != 0) || _la==DecimalNumber || _la==VersionLiteral) {
				{
				{
				setState(220);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__2) {
					{
					setState(219);
					match(T__2);
					}
				}

				setState(222);
				versionConstraint();
				}
				}
				setState(227);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VersionOperatorContext extends ParserRuleContext {
		public VersionOperatorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_versionOperator; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVersionOperator(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVersionOperator(this);
		}
	}

	public final VersionOperatorContext versionOperator() throws RecognitionException {
		VersionOperatorContext _localctx = new VersionOperatorContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_versionOperator);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(228);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 2032L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VersionConstraintContext extends ParserRuleContext {
		public TerminalNode VersionLiteral() { return getToken(SolidityParser.VersionLiteral, 0); }
		public VersionOperatorContext versionOperator() {
			return getRuleContext(VersionOperatorContext.class,0);
		}
		public TerminalNode DecimalNumber() { return getToken(SolidityParser.DecimalNumber, 0); }
		public VersionConstraintContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_versionConstraint; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVersionConstraint(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVersionConstraint(this);
		}
	}

	public final VersionConstraintContext versionConstraint() throws RecognitionException {
		VersionConstraintContext _localctx = new VersionConstraintContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_versionConstraint);
		int _la;
		try {
			setState(238);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(231);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2032L) != 0)) {
					{
					setState(230);
					versionOperator();
					}
				}

				setState(233);
				match(VersionLiteral);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(235);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2032L) != 0)) {
					{
					setState(234);
					versionOperator();
					}
				}

				setState(237);
				match(DecimalNumber);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ImportDeclarationContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public ImportDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_importDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterImportDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitImportDeclaration(this);
		}
	}

	public final ImportDeclarationContext importDeclaration() throws RecognitionException {
		ImportDeclarationContext _localctx = new ImportDeclarationContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_importDeclaration);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(240);
			identifier();
			setState(243);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__10) {
				{
				setState(241);
				match(T__10);
				setState(242);
				identifier();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ImportDirectiveContext extends ParserRuleContext {
		public ImportPathContext importPath() {
			return getRuleContext(ImportPathContext.class,0);
		}
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public List<ImportDeclarationContext> importDeclaration() {
			return getRuleContexts(ImportDeclarationContext.class);
		}
		public ImportDeclarationContext importDeclaration(int i) {
			return getRuleContext(ImportDeclarationContext.class,i);
		}
		public ImportDirectiveContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_importDirective; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterImportDirective(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitImportDirective(this);
		}
	}

	public final ImportDirectiveContext importDirective() throws RecognitionException {
		ImportDirectiveContext _localctx = new ImportDirectiveContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_importDirective);
		int _la;
		try {
			setState(281);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,13,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(245);
				match(T__11);
				setState(246);
				importPath();
				setState(249);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__10) {
					{
					setState(247);
					match(T__10);
					setState(248);
					identifier();
					}
				}

				setState(251);
				match(T__1);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(253);
				match(T__11);
				setState(256);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case T__12:
					{
					setState(254);
					match(T__12);
					}
					break;
				case T__13:
				case T__24:
				case T__41:
				case T__53:
				case T__95:
				case LeaveKeyword:
				case PayableKeyword:
				case ConstructorKeyword:
				case ReceiveKeyword:
				case Identifier:
					{
					setState(255);
					identifier();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(260);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__10) {
					{
					setState(258);
					match(T__10);
					setState(259);
					identifier();
					}
				}

				setState(262);
				match(T__13);
				setState(263);
				importPath();
				setState(264);
				match(T__1);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(266);
				match(T__11);
				setState(267);
				match(T__14);
				setState(268);
				importDeclaration();
				setState(273);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(269);
					match(T__15);
					setState(270);
					importDeclaration();
					}
					}
					setState(275);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(276);
				match(T__16);
				setState(277);
				match(T__13);
				setState(278);
				importPath();
				setState(279);
				match(T__1);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ImportPathContext extends ParserRuleContext {
		public TerminalNode StringLiteralFragment() { return getToken(SolidityParser.StringLiteralFragment, 0); }
		public ImportPathContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_importPath; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterImportPath(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitImportPath(this);
		}
	}

	public final ImportPathContext importPath() throws RecognitionException {
		ImportPathContext _localctx = new ImportPathContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_importPath);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(283);
			match(StringLiteralFragment);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ContractDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<InheritanceSpecifierContext> inheritanceSpecifier() {
			return getRuleContexts(InheritanceSpecifierContext.class);
		}
		public InheritanceSpecifierContext inheritanceSpecifier(int i) {
			return getRuleContext(InheritanceSpecifierContext.class,i);
		}
		public List<ContractPartContext> contractPart() {
			return getRuleContexts(ContractPartContext.class);
		}
		public ContractPartContext contractPart(int i) {
			return getRuleContext(ContractPartContext.class,i);
		}
		public ContractDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_contractDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterContractDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitContractDefinition(this);
		}
	}

	public final ContractDefinitionContext contractDefinition() throws RecognitionException {
		ContractDefinitionContext _localctx = new ContractDefinitionContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_contractDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(286);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__17) {
				{
				setState(285);
				match(T__17);
				}
			}

			setState(288);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 3670016L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			setState(289);
			identifier();
			setState(299);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__21) {
				{
				setState(290);
				match(T__21);
				setState(291);
				inheritanceSpecifier();
				setState(296);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(292);
					match(T__15);
					setState(293);
					inheritanceSpecifier();
					}
					}
					setState(298);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(301);
			match(T__14);
			setState(305);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451110302466048L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 8057520253L) != 0)) {
				{
				{
				setState(302);
				contractPart();
				}
				}
				setState(307);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(308);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class InheritanceSpecifierContext extends ParserRuleContext {
		public UserDefinedTypeNameContext userDefinedTypeName() {
			return getRuleContext(UserDefinedTypeNameContext.class,0);
		}
		public ExpressionListContext expressionList() {
			return getRuleContext(ExpressionListContext.class,0);
		}
		public InheritanceSpecifierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_inheritanceSpecifier; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterInheritanceSpecifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitInheritanceSpecifier(this);
		}
	}

	public final InheritanceSpecifierContext inheritanceSpecifier() throws RecognitionException {
		InheritanceSpecifierContext _localctx = new InheritanceSpecifierContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_inheritanceSpecifier);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(310);
			userDefinedTypeName();
			setState(316);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__22) {
				{
				setState(311);
				match(T__22);
				setState(313);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
					{
					setState(312);
					expressionList();
					}
				}

				setState(315);
				match(T__23);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ContractPartContext extends ParserRuleContext {
		public StateVariableDeclarationContext stateVariableDeclaration() {
			return getRuleContext(StateVariableDeclarationContext.class,0);
		}
		public UsingForDeclarationContext usingForDeclaration() {
			return getRuleContext(UsingForDeclarationContext.class,0);
		}
		public StructDefinitionContext structDefinition() {
			return getRuleContext(StructDefinitionContext.class,0);
		}
		public ModifierDefinitionContext modifierDefinition() {
			return getRuleContext(ModifierDefinitionContext.class,0);
		}
		public FunctionDefinitionContext functionDefinition() {
			return getRuleContext(FunctionDefinitionContext.class,0);
		}
		public EventDefinitionContext eventDefinition() {
			return getRuleContext(EventDefinitionContext.class,0);
		}
		public EnumDefinitionContext enumDefinition() {
			return getRuleContext(EnumDefinitionContext.class,0);
		}
		public CustomErrorDefinitionContext customErrorDefinition() {
			return getRuleContext(CustomErrorDefinitionContext.class,0);
		}
		public ContractPartContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_contractPart; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterContractPart(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitContractPart(this);
		}
	}

	public final ContractPartContext contractPart() throws RecognitionException {
		ContractPartContext _localctx = new ContractPartContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_contractPart);
		try {
			setState(326);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,20,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(318);
				stateVariableDeclaration();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(319);
				usingForDeclaration();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(320);
				structDefinition();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(321);
				modifierDefinition();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(322);
				functionDefinition();
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(323);
				eventDefinition();
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(324);
				enumDefinition();
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(325);
				customErrorDefinition();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StateVariableDeclarationContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<TerminalNode> PublicKeyword() { return getTokens(SolidityParser.PublicKeyword); }
		public TerminalNode PublicKeyword(int i) {
			return getToken(SolidityParser.PublicKeyword, i);
		}
		public List<TerminalNode> InternalKeyword() { return getTokens(SolidityParser.InternalKeyword); }
		public TerminalNode InternalKeyword(int i) {
			return getToken(SolidityParser.InternalKeyword, i);
		}
		public List<TerminalNode> PrivateKeyword() { return getTokens(SolidityParser.PrivateKeyword); }
		public TerminalNode PrivateKeyword(int i) {
			return getToken(SolidityParser.PrivateKeyword, i);
		}
		public List<TerminalNode> ConstantKeyword() { return getTokens(SolidityParser.ConstantKeyword); }
		public TerminalNode ConstantKeyword(int i) {
			return getToken(SolidityParser.ConstantKeyword, i);
		}
		public List<TerminalNode> ImmutableKeyword() { return getTokens(SolidityParser.ImmutableKeyword); }
		public TerminalNode ImmutableKeyword(int i) {
			return getToken(SolidityParser.ImmutableKeyword, i);
		}
		public List<OverrideSpecifierContext> overrideSpecifier() {
			return getRuleContexts(OverrideSpecifierContext.class);
		}
		public OverrideSpecifierContext overrideSpecifier(int i) {
			return getRuleContext(OverrideSpecifierContext.class,i);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public StateVariableDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_stateVariableDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStateVariableDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStateVariableDeclaration(this);
		}
	}

	public final StateVariableDeclarationContext stateVariableDeclaration() throws RecognitionException {
		StateVariableDeclarationContext _localctx = new StateVariableDeclarationContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_stateVariableDeclaration);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(328);
			typeName(0);
			setState(337);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (((((_la - 97)) & ~0x3f) == 0 && ((1L << (_la - 97)) & 13680641L) != 0)) {
				{
				setState(335);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case PublicKeyword:
					{
					setState(329);
					match(PublicKeyword);
					}
					break;
				case InternalKeyword:
					{
					setState(330);
					match(InternalKeyword);
					}
					break;
				case PrivateKeyword:
					{
					setState(331);
					match(PrivateKeyword);
					}
					break;
				case ConstantKeyword:
					{
					setState(332);
					match(ConstantKeyword);
					}
					break;
				case ImmutableKeyword:
					{
					setState(333);
					match(ImmutableKeyword);
					}
					break;
				case T__96:
					{
					setState(334);
					overrideSpecifier();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				}
				setState(339);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(340);
			identifier();
			setState(343);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__9) {
				{
				setState(341);
				match(T__9);
				setState(342);
				expression(0);
				}
			}

			setState(345);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FileLevelConstantContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public TerminalNode ConstantKeyword() { return getToken(SolidityParser.ConstantKeyword, 0); }
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public FileLevelConstantContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_fileLevelConstant; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFileLevelConstant(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFileLevelConstant(this);
		}
	}

	public final FileLevelConstantContext fileLevelConstant() throws RecognitionException {
		FileLevelConstantContext _localctx = new FileLevelConstantContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_fileLevelConstant);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(347);
			typeName(0);
			setState(348);
			match(ConstantKeyword);
			setState(349);
			identifier();
			setState(350);
			match(T__9);
			setState(351);
			expression(0);
			setState(352);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class CustomErrorDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public ParameterListContext parameterList() {
			return getRuleContext(ParameterListContext.class,0);
		}
		public CustomErrorDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_customErrorDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterCustomErrorDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitCustomErrorDefinition(this);
		}
	}

	public final CustomErrorDefinitionContext customErrorDefinition() throws RecognitionException {
		CustomErrorDefinitionContext _localctx = new CustomErrorDefinitionContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_customErrorDefinition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(354);
			match(T__24);
			setState(355);
			identifier();
			setState(356);
			parameterList();
			setState(357);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class UsingForDeclarationContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public UsingForDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_usingForDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterUsingForDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitUsingForDeclaration(this);
		}
	}

	public final UsingForDeclarationContext usingForDeclaration() throws RecognitionException {
		UsingForDeclarationContext _localctx = new UsingForDeclarationContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_usingForDeclaration);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(359);
			match(T__25);
			setState(360);
			identifier();
			setState(361);
			match(T__26);
			setState(364);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__12:
				{
				setState(362);
				match(T__12);
				}
				break;
			case T__13:
			case T__24:
			case T__29:
			case T__35:
			case T__37:
			case T__41:
			case T__53:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case T__95:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
				{
				setState(363);
				typeName(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(366);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StructDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<VariableDeclarationContext> variableDeclaration() {
			return getRuleContexts(VariableDeclarationContext.class);
		}
		public VariableDeclarationContext variableDeclaration(int i) {
			return getRuleContext(VariableDeclarationContext.class,i);
		}
		public StructDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_structDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStructDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStructDefinition(this);
		}
	}

	public final StructDefinitionContext structDefinition() throws RecognitionException {
		StructDefinitionContext _localctx = new StructDefinitionContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_structDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(368);
			match(T__27);
			setState(369);
			identifier();
			setState(370);
			match(T__14);
			setState(381);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
				{
				setState(371);
				variableDeclaration();
				setState(372);
				match(T__1);
				setState(378);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
					{
					{
					setState(373);
					variableDeclaration();
					setState(374);
					match(T__1);
					}
					}
					setState(380);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(383);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ModifierDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public ParameterListContext parameterList() {
			return getRuleContext(ParameterListContext.class,0);
		}
		public List<TerminalNode> VirtualKeyword() { return getTokens(SolidityParser.VirtualKeyword); }
		public TerminalNode VirtualKeyword(int i) {
			return getToken(SolidityParser.VirtualKeyword, i);
		}
		public List<OverrideSpecifierContext> overrideSpecifier() {
			return getRuleContexts(OverrideSpecifierContext.class);
		}
		public OverrideSpecifierContext overrideSpecifier(int i) {
			return getRuleContext(OverrideSpecifierContext.class,i);
		}
		public ModifierDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_modifierDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterModifierDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitModifierDefinition(this);
		}
	}

	public final ModifierDefinitionContext modifierDefinition() throws RecognitionException {
		ModifierDefinitionContext _localctx = new ModifierDefinitionContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_modifierDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(385);
			match(T__28);
			setState(386);
			identifier();
			setState(388);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__22) {
				{
				setState(387);
				parameterList();
				}
			}

			setState(394);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__96 || _la==VirtualKeyword) {
				{
				setState(392);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case VirtualKeyword:
					{
					setState(390);
					match(VirtualKeyword);
					}
					break;
				case T__96:
					{
					setState(391);
					overrideSpecifier();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				}
				setState(396);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(399);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__1:
				{
				setState(397);
				match(T__1);
				}
				break;
			case T__14:
				{
				setState(398);
				block();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ModifierInvocationContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public ExpressionListContext expressionList() {
			return getRuleContext(ExpressionListContext.class,0);
		}
		public ModifierInvocationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_modifierInvocation; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterModifierInvocation(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitModifierInvocation(this);
		}
	}

	public final ModifierInvocationContext modifierInvocation() throws RecognitionException {
		ModifierInvocationContext _localctx = new ModifierInvocationContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_modifierInvocation);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(401);
			identifier();
			setState(407);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__22) {
				{
				setState(402);
				match(T__22);
				setState(404);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
					{
					setState(403);
					expressionList();
					}
				}

				setState(406);
				match(T__23);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionDefinitionContext extends ParserRuleContext {
		public FunctionDescriptorContext functionDescriptor() {
			return getRuleContext(FunctionDescriptorContext.class,0);
		}
		public ParameterListContext parameterList() {
			return getRuleContext(ParameterListContext.class,0);
		}
		public ModifierListContext modifierList() {
			return getRuleContext(ModifierListContext.class,0);
		}
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public ReturnParametersContext returnParameters() {
			return getRuleContext(ReturnParametersContext.class,0);
		}
		public FunctionDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionDefinition(this);
		}
	}

	public final FunctionDefinitionContext functionDefinition() throws RecognitionException {
		FunctionDefinitionContext _localctx = new FunctionDefinitionContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_functionDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(409);
			functionDescriptor();
			setState(410);
			parameterList();
			setState(411);
			modifierList();
			setState(413);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__30) {
				{
				setState(412);
				returnParameters();
				}
			}

			setState(417);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__1:
				{
				setState(415);
				match(T__1);
				}
				break;
			case T__14:
				{
				setState(416);
				block();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionDescriptorContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TerminalNode ConstructorKeyword() { return getToken(SolidityParser.ConstructorKeyword, 0); }
		public TerminalNode FallbackKeyword() { return getToken(SolidityParser.FallbackKeyword, 0); }
		public TerminalNode ReceiveKeyword() { return getToken(SolidityParser.ReceiveKeyword, 0); }
		public FunctionDescriptorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionDescriptor; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionDescriptor(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionDescriptor(this);
		}
	}

	public final FunctionDescriptorContext functionDescriptor() throws RecognitionException {
		FunctionDescriptorContext _localctx = new FunctionDescriptorContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_functionDescriptor);
		int _la;
		try {
			setState(426);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__29:
				enterOuterAlt(_localctx, 1);
				{
				setState(419);
				match(T__29);
				setState(421);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
					{
					setState(420);
					identifier();
					}
				}

				}
				break;
			case ConstructorKeyword:
				enterOuterAlt(_localctx, 2);
				{
				setState(423);
				match(ConstructorKeyword);
				}
				break;
			case FallbackKeyword:
				enterOuterAlt(_localctx, 3);
				{
				setState(424);
				match(FallbackKeyword);
				}
				break;
			case ReceiveKeyword:
				enterOuterAlt(_localctx, 4);
				{
				setState(425);
				match(ReceiveKeyword);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ReturnParametersContext extends ParserRuleContext {
		public ParameterListContext parameterList() {
			return getRuleContext(ParameterListContext.class,0);
		}
		public ReturnParametersContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_returnParameters; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterReturnParameters(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitReturnParameters(this);
		}
	}

	public final ReturnParametersContext returnParameters() throws RecognitionException {
		ReturnParametersContext _localctx = new ReturnParametersContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_returnParameters);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(428);
			match(T__30);
			setState(429);
			parameterList();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ModifierListContext extends ParserRuleContext {
		public List<TerminalNode> ExternalKeyword() { return getTokens(SolidityParser.ExternalKeyword); }
		public TerminalNode ExternalKeyword(int i) {
			return getToken(SolidityParser.ExternalKeyword, i);
		}
		public List<TerminalNode> PublicKeyword() { return getTokens(SolidityParser.PublicKeyword); }
		public TerminalNode PublicKeyword(int i) {
			return getToken(SolidityParser.PublicKeyword, i);
		}
		public List<TerminalNode> InternalKeyword() { return getTokens(SolidityParser.InternalKeyword); }
		public TerminalNode InternalKeyword(int i) {
			return getToken(SolidityParser.InternalKeyword, i);
		}
		public List<TerminalNode> PrivateKeyword() { return getTokens(SolidityParser.PrivateKeyword); }
		public TerminalNode PrivateKeyword(int i) {
			return getToken(SolidityParser.PrivateKeyword, i);
		}
		public List<TerminalNode> VirtualKeyword() { return getTokens(SolidityParser.VirtualKeyword); }
		public TerminalNode VirtualKeyword(int i) {
			return getToken(SolidityParser.VirtualKeyword, i);
		}
		public List<StateMutabilityContext> stateMutability() {
			return getRuleContexts(StateMutabilityContext.class);
		}
		public StateMutabilityContext stateMutability(int i) {
			return getRuleContext(StateMutabilityContext.class,i);
		}
		public List<ModifierInvocationContext> modifierInvocation() {
			return getRuleContexts(ModifierInvocationContext.class);
		}
		public ModifierInvocationContext modifierInvocation(int i) {
			return getRuleContext(ModifierInvocationContext.class,i);
		}
		public List<OverrideSpecifierContext> overrideSpecifier() {
			return getRuleContexts(OverrideSpecifierContext.class);
		}
		public OverrideSpecifierContext overrideSpecifier(int i) {
			return getRuleContext(OverrideSpecifierContext.class,i);
		}
		public ModifierListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_modifierList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterModifierList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitModifierList(this);
		}
	}

	public final ModifierListContext modifierList() throws RecognitionException {
		ModifierListContext _localctx = new ModifierListContext(_ctx, getState());
		enterRule(_localctx, 46, RULE_modifierList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(441);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 7380697091L) != 0)) {
				{
				setState(439);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,37,_ctx) ) {
				case 1:
					{
					setState(431);
					match(ExternalKeyword);
					}
					break;
				case 2:
					{
					setState(432);
					match(PublicKeyword);
					}
					break;
				case 3:
					{
					setState(433);
					match(InternalKeyword);
					}
					break;
				case 4:
					{
					setState(434);
					match(PrivateKeyword);
					}
					break;
				case 5:
					{
					setState(435);
					match(VirtualKeyword);
					}
					break;
				case 6:
					{
					setState(436);
					stateMutability();
					}
					break;
				case 7:
					{
					setState(437);
					modifierInvocation();
					}
					break;
				case 8:
					{
					setState(438);
					overrideSpecifier();
					}
					break;
				}
				}
				setState(443);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EventDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public EventParameterListContext eventParameterList() {
			return getRuleContext(EventParameterListContext.class,0);
		}
		public TerminalNode AnonymousKeyword() { return getToken(SolidityParser.AnonymousKeyword, 0); }
		public EventDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_eventDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEventDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEventDefinition(this);
		}
	}

	public final EventDefinitionContext eventDefinition() throws RecognitionException {
		EventDefinitionContext _localctx = new EventDefinitionContext(_ctx, getState());
		enterRule(_localctx, 48, RULE_eventDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(444);
			match(T__31);
			setState(445);
			identifier();
			setState(446);
			eventParameterList();
			setState(448);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==AnonymousKeyword) {
				{
				setState(447);
				match(AnonymousKeyword);
				}
			}

			setState(450);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EnumValueContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public EnumValueContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_enumValue; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEnumValue(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEnumValue(this);
		}
	}

	public final EnumValueContext enumValue() throws RecognitionException {
		EnumValueContext _localctx = new EnumValueContext(_ctx, getState());
		enterRule(_localctx, 50, RULE_enumValue);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(452);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EnumDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<EnumValueContext> enumValue() {
			return getRuleContexts(EnumValueContext.class);
		}
		public EnumValueContext enumValue(int i) {
			return getRuleContext(EnumValueContext.class,i);
		}
		public EnumDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_enumDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEnumDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEnumDefinition(this);
		}
	}

	public final EnumDefinitionContext enumDefinition() throws RecognitionException {
		EnumDefinitionContext _localctx = new EnumDefinitionContext(_ctx, getState());
		enterRule(_localctx, 52, RULE_enumDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(454);
			match(T__32);
			setState(455);
			identifier();
			setState(456);
			match(T__14);
			setState(458);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(457);
				enumValue();
				}
			}

			setState(464);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__15) {
				{
				{
				setState(460);
				match(T__15);
				setState(461);
				enumValue();
				}
				}
				setState(466);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(467);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ParameterListContext extends ParserRuleContext {
		public List<ParameterContext> parameter() {
			return getRuleContexts(ParameterContext.class);
		}
		public ParameterContext parameter(int i) {
			return getRuleContext(ParameterContext.class,i);
		}
		public ParameterListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_parameterList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterParameterList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitParameterList(this);
		}
	}

	public final ParameterListContext parameterList() throws RecognitionException {
		ParameterListContext _localctx = new ParameterListContext(_ctx, getState());
		enterRule(_localctx, 54, RULE_parameterList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(469);
			match(T__22);
			setState(478);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
				{
				setState(470);
				parameter();
				setState(475);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(471);
					match(T__15);
					setState(472);
					parameter();
					}
					}
					setState(477);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(480);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ParameterContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public StorageLocationContext storageLocation() {
			return getRuleContext(StorageLocationContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public ParameterContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_parameter; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterParameter(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitParameter(this);
		}
	}

	public final ParameterContext parameter() throws RecognitionException {
		ParameterContext _localctx = new ParameterContext(_ctx, getState());
		enterRule(_localctx, 56, RULE_parameter);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(482);
			typeName(0);
			setState(484);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,44,_ctx) ) {
			case 1:
				{
				setState(483);
				storageLocation();
				}
				break;
			}
			setState(487);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(486);
				identifier();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EventParameterListContext extends ParserRuleContext {
		public List<EventParameterContext> eventParameter() {
			return getRuleContexts(EventParameterContext.class);
		}
		public EventParameterContext eventParameter(int i) {
			return getRuleContext(EventParameterContext.class,i);
		}
		public EventParameterListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_eventParameterList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEventParameterList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEventParameterList(this);
		}
	}

	public final EventParameterListContext eventParameterList() throws RecognitionException {
		EventParameterListContext _localctx = new EventParameterListContext(_ctx, getState());
		enterRule(_localctx, 58, RULE_eventParameterList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(489);
			match(T__22);
			setState(498);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
				{
				setState(490);
				eventParameter();
				setState(495);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(491);
					match(T__15);
					setState(492);
					eventParameter();
					}
					}
					setState(497);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(500);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EventParameterContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public TerminalNode IndexedKeyword() { return getToken(SolidityParser.IndexedKeyword, 0); }
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public EventParameterContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_eventParameter; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEventParameter(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEventParameter(this);
		}
	}

	public final EventParameterContext eventParameter() throws RecognitionException {
		EventParameterContext _localctx = new EventParameterContext(_ctx, getState());
		enterRule(_localctx, 60, RULE_eventParameter);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(502);
			typeName(0);
			setState(504);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==IndexedKeyword) {
				{
				setState(503);
				match(IndexedKeyword);
				}
			}

			setState(507);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(506);
				identifier();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionTypeParameterListContext extends ParserRuleContext {
		public List<FunctionTypeParameterContext> functionTypeParameter() {
			return getRuleContexts(FunctionTypeParameterContext.class);
		}
		public FunctionTypeParameterContext functionTypeParameter(int i) {
			return getRuleContext(FunctionTypeParameterContext.class,i);
		}
		public FunctionTypeParameterListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionTypeParameterList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionTypeParameterList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionTypeParameterList(this);
		}
	}

	public final FunctionTypeParameterListContext functionTypeParameterList() throws RecognitionException {
		FunctionTypeParameterListContext _localctx = new FunctionTypeParameterListContext(_ctx, getState());
		enterRule(_localctx, 62, RULE_functionTypeParameterList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(509);
			match(T__22);
			setState(518);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
				{
				setState(510);
				functionTypeParameter();
				setState(515);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(511);
					match(T__15);
					setState(512);
					functionTypeParameter();
					}
					}
					setState(517);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(520);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionTypeParameterContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public StorageLocationContext storageLocation() {
			return getRuleContext(StorageLocationContext.class,0);
		}
		public FunctionTypeParameterContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionTypeParameter; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionTypeParameter(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionTypeParameter(this);
		}
	}

	public final FunctionTypeParameterContext functionTypeParameter() throws RecognitionException {
		FunctionTypeParameterContext _localctx = new FunctionTypeParameterContext(_ctx, getState());
		enterRule(_localctx, 64, RULE_functionTypeParameter);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(522);
			typeName(0);
			setState(524);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7696581394432L) != 0)) {
				{
				setState(523);
				storageLocation();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VariableDeclarationContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public StorageLocationContext storageLocation() {
			return getRuleContext(StorageLocationContext.class,0);
		}
		public VariableDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variableDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVariableDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVariableDeclaration(this);
		}
	}

	public final VariableDeclarationContext variableDeclaration() throws RecognitionException {
		VariableDeclarationContext _localctx = new VariableDeclarationContext(_ctx, getState());
		enterRule(_localctx, 66, RULE_variableDeclaration);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(526);
			typeName(0);
			setState(528);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,53,_ctx) ) {
			case 1:
				{
				setState(527);
				storageLocation();
				}
				break;
			}
			setState(530);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TypeNameContext extends ParserRuleContext {
		public ElementaryTypeNameContext elementaryTypeName() {
			return getRuleContext(ElementaryTypeNameContext.class,0);
		}
		public UserDefinedTypeNameContext userDefinedTypeName() {
			return getRuleContext(UserDefinedTypeNameContext.class,0);
		}
		public MappingContext mapping() {
			return getRuleContext(MappingContext.class,0);
		}
		public FunctionTypeNameContext functionTypeName() {
			return getRuleContext(FunctionTypeNameContext.class,0);
		}
		public TerminalNode PayableKeyword() { return getToken(SolidityParser.PayableKeyword, 0); }
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public TypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterTypeName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitTypeName(this);
		}
	}

	public final TypeNameContext typeName() throws RecognitionException {
		return typeName(0);
	}

	private TypeNameContext typeName(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		TypeNameContext _localctx = new TypeNameContext(_ctx, _parentState);
		TypeNameContext _prevctx = _localctx;
		int _startState = 68;
		enterRecursionRule(_localctx, 68, RULE_typeName, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(539);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,54,_ctx) ) {
			case 1:
				{
				setState(533);
				elementaryTypeName();
				}
				break;
			case 2:
				{
				setState(534);
				userDefinedTypeName();
				}
				break;
			case 3:
				{
				setState(535);
				mapping();
				}
				break;
			case 4:
				{
				setState(536);
				functionTypeName();
				}
				break;
			case 5:
				{
				setState(537);
				match(T__35);
				setState(538);
				match(PayableKeyword);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(549);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,56,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new TypeNameContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_typeName);
					setState(541);
					if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
					setState(542);
					match(T__33);
					setState(544);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
						{
						setState(543);
						expression(0);
						}
					}

					setState(546);
					match(T__34);
					}
					} 
				}
				setState(551);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,56,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class UserDefinedTypeNameContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public UserDefinedTypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_userDefinedTypeName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterUserDefinedTypeName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitUserDefinedTypeName(this);
		}
	}

	public final UserDefinedTypeNameContext userDefinedTypeName() throws RecognitionException {
		UserDefinedTypeNameContext _localctx = new UserDefinedTypeNameContext(_ctx, getState());
		enterRule(_localctx, 70, RULE_userDefinedTypeName);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(552);
			identifier();
			setState(557);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,57,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(553);
					match(T__36);
					setState(554);
					identifier();
					}
					} 
				}
				setState(559);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,57,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MappingKeyContext extends ParserRuleContext {
		public ElementaryTypeNameContext elementaryTypeName() {
			return getRuleContext(ElementaryTypeNameContext.class,0);
		}
		public UserDefinedTypeNameContext userDefinedTypeName() {
			return getRuleContext(UserDefinedTypeNameContext.class,0);
		}
		public MappingKeyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mappingKey; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterMappingKey(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitMappingKey(this);
		}
	}

	public final MappingKeyContext mappingKey() throws RecognitionException {
		MappingKeyContext _localctx = new MappingKeyContext(_ctx, getState());
		enterRule(_localctx, 72, RULE_mappingKey);
		try {
			setState(562);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__35:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
				enterOuterAlt(_localctx, 1);
				{
				setState(560);
				elementaryTypeName();
				}
				break;
			case T__13:
			case T__24:
			case T__41:
			case T__53:
			case T__95:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
				enterOuterAlt(_localctx, 2);
				{
				setState(561);
				userDefinedTypeName();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MappingContext extends ParserRuleContext {
		public MappingKeyContext mappingKey() {
			return getRuleContext(MappingKeyContext.class,0);
		}
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public MappingContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mapping; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterMapping(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitMapping(this);
		}
	}

	public final MappingContext mapping() throws RecognitionException {
		MappingContext _localctx = new MappingContext(_ctx, getState());
		enterRule(_localctx, 74, RULE_mapping);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(564);
			match(T__37);
			setState(565);
			match(T__22);
			setState(566);
			mappingKey();
			setState(567);
			match(T__38);
			setState(568);
			typeName(0);
			setState(569);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionTypeNameContext extends ParserRuleContext {
		public List<FunctionTypeParameterListContext> functionTypeParameterList() {
			return getRuleContexts(FunctionTypeParameterListContext.class);
		}
		public FunctionTypeParameterListContext functionTypeParameterList(int i) {
			return getRuleContext(FunctionTypeParameterListContext.class,i);
		}
		public List<TerminalNode> InternalKeyword() { return getTokens(SolidityParser.InternalKeyword); }
		public TerminalNode InternalKeyword(int i) {
			return getToken(SolidityParser.InternalKeyword, i);
		}
		public List<TerminalNode> ExternalKeyword() { return getTokens(SolidityParser.ExternalKeyword); }
		public TerminalNode ExternalKeyword(int i) {
			return getToken(SolidityParser.ExternalKeyword, i);
		}
		public List<StateMutabilityContext> stateMutability() {
			return getRuleContexts(StateMutabilityContext.class);
		}
		public StateMutabilityContext stateMutability(int i) {
			return getRuleContext(StateMutabilityContext.class,i);
		}
		public FunctionTypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionTypeName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionTypeName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionTypeName(this);
		}
	}

	public final FunctionTypeNameContext functionTypeName() throws RecognitionException {
		FunctionTypeNameContext _localctx = new FunctionTypeNameContext(_ctx, getState());
		enterRule(_localctx, 76, RULE_functionTypeName);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(571);
			match(T__29);
			setState(572);
			functionTypeParameterList();
			setState(578);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,60,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					setState(576);
					_errHandler.sync(this);
					switch (_input.LA(1)) {
					case InternalKeyword:
						{
						setState(573);
						match(InternalKeyword);
						}
						break;
					case ExternalKeyword:
						{
						setState(574);
						match(ExternalKeyword);
						}
						break;
					case ConstantKeyword:
					case PayableKeyword:
					case PureKeyword:
					case ViewKeyword:
						{
						setState(575);
						stateMutability();
						}
						break;
					default:
						throw new NoViableAltException(this);
					}
					} 
				}
				setState(580);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,60,_ctx);
			}
			setState(583);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,61,_ctx) ) {
			case 1:
				{
				setState(581);
				match(T__30);
				setState(582);
				functionTypeParameterList();
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StorageLocationContext extends ParserRuleContext {
		public StorageLocationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_storageLocation; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStorageLocation(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStorageLocation(this);
		}
	}

	public final StorageLocationContext storageLocation() throws RecognitionException {
		StorageLocationContext _localctx = new StorageLocationContext(_ctx, getState());
		enterRule(_localctx, 78, RULE_storageLocation);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(585);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 7696581394432L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StateMutabilityContext extends ParserRuleContext {
		public TerminalNode PureKeyword() { return getToken(SolidityParser.PureKeyword, 0); }
		public TerminalNode ConstantKeyword() { return getToken(SolidityParser.ConstantKeyword, 0); }
		public TerminalNode ViewKeyword() { return getToken(SolidityParser.ViewKeyword, 0); }
		public TerminalNode PayableKeyword() { return getToken(SolidityParser.PayableKeyword, 0); }
		public StateMutabilityContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_stateMutability; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStateMutability(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStateMutability(this);
		}
	}

	public final StateMutabilityContext stateMutability() throws RecognitionException {
		StateMutabilityContext _localctx = new StateMutabilityContext(_ctx, getState());
		enterRule(_localctx, 80, RULE_stateMutability);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(587);
			_la = _input.LA(1);
			if ( !(((((_la - 111)) & ~0x3f) == 0 && ((1L << (_la - 111)) & 10369L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BlockContext extends ParserRuleContext {
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public BlockContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_block; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterBlock(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitBlock(this);
		}
	}

	public final BlockContext block() throws RecognitionException {
		BlockContext _localctx = new BlockContext(_ctx, getState());
		enterRule(_localctx, 82, RULE_block);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(589);
			match(T__14);
			setState(593);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9079253973874247169L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15708081149L) != 0)) {
				{
				{
				setState(590);
				statement();
				}
				}
				setState(595);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(596);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StatementContext extends ParserRuleContext {
		public IfStatementContext ifStatement() {
			return getRuleContext(IfStatementContext.class,0);
		}
		public TryStatementContext tryStatement() {
			return getRuleContext(TryStatementContext.class,0);
		}
		public WhileStatementContext whileStatement() {
			return getRuleContext(WhileStatementContext.class,0);
		}
		public ForStatementContext forStatement() {
			return getRuleContext(ForStatementContext.class,0);
		}
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public InlineAssemblyStatementContext inlineAssemblyStatement() {
			return getRuleContext(InlineAssemblyStatementContext.class,0);
		}
		public DoWhileStatementContext doWhileStatement() {
			return getRuleContext(DoWhileStatementContext.class,0);
		}
		public ContinueStatementContext continueStatement() {
			return getRuleContext(ContinueStatementContext.class,0);
		}
		public BreakStatementContext breakStatement() {
			return getRuleContext(BreakStatementContext.class,0);
		}
		public ReturnStatementContext returnStatement() {
			return getRuleContext(ReturnStatementContext.class,0);
		}
		public ThrowStatementContext throwStatement() {
			return getRuleContext(ThrowStatementContext.class,0);
		}
		public EmitStatementContext emitStatement() {
			return getRuleContext(EmitStatementContext.class,0);
		}
		public SimpleStatementContext simpleStatement() {
			return getRuleContext(SimpleStatementContext.class,0);
		}
		public UncheckedStatementContext uncheckedStatement() {
			return getRuleContext(UncheckedStatementContext.class,0);
		}
		public RevertStatementContext revertStatement() {
			return getRuleContext(RevertStatementContext.class,0);
		}
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStatement(this);
		}
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 84, RULE_statement);
		try {
			setState(613);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,63,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(598);
				ifStatement();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(599);
				tryStatement();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(600);
				whileStatement();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(601);
				forStatement();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(602);
				block();
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(603);
				inlineAssemblyStatement();
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(604);
				doWhileStatement();
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(605);
				continueStatement();
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(606);
				breakStatement();
				}
				break;
			case 10:
				enterOuterAlt(_localctx, 10);
				{
				setState(607);
				returnStatement();
				}
				break;
			case 11:
				enterOuterAlt(_localctx, 11);
				{
				setState(608);
				throwStatement();
				}
				break;
			case 12:
				enterOuterAlt(_localctx, 12);
				{
				setState(609);
				emitStatement();
				}
				break;
			case 13:
				enterOuterAlt(_localctx, 13);
				{
				setState(610);
				simpleStatement();
				}
				break;
			case 14:
				enterOuterAlt(_localctx, 14);
				{
				setState(611);
				uncheckedStatement();
				}
				break;
			case 15:
				enterOuterAlt(_localctx, 15);
				{
				setState(612);
				revertStatement();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionStatementContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public ExpressionStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expressionStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterExpressionStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitExpressionStatement(this);
		}
	}

	public final ExpressionStatementContext expressionStatement() throws RecognitionException {
		ExpressionStatementContext _localctx = new ExpressionStatementContext(_ctx, getState());
		enterRule(_localctx, 86, RULE_expressionStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(615);
			expression(0);
			setState(616);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IfStatementContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public IfStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ifStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterIfStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitIfStatement(this);
		}
	}

	public final IfStatementContext ifStatement() throws RecognitionException {
		IfStatementContext _localctx = new IfStatementContext(_ctx, getState());
		enterRule(_localctx, 88, RULE_ifStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(618);
			match(T__42);
			setState(619);
			match(T__22);
			setState(620);
			expression(0);
			setState(621);
			match(T__23);
			setState(622);
			statement();
			setState(625);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,64,_ctx) ) {
			case 1:
				{
				setState(623);
				match(T__43);
				setState(624);
				statement();
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TryStatementContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public ReturnParametersContext returnParameters() {
			return getRuleContext(ReturnParametersContext.class,0);
		}
		public List<CatchClauseContext> catchClause() {
			return getRuleContexts(CatchClauseContext.class);
		}
		public CatchClauseContext catchClause(int i) {
			return getRuleContext(CatchClauseContext.class,i);
		}
		public TryStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_tryStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterTryStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitTryStatement(this);
		}
	}

	public final TryStatementContext tryStatement() throws RecognitionException {
		TryStatementContext _localctx = new TryStatementContext(_ctx, getState());
		enterRule(_localctx, 90, RULE_tryStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(627);
			match(T__44);
			setState(628);
			expression(0);
			setState(630);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__30) {
				{
				setState(629);
				returnParameters();
				}
			}

			setState(632);
			block();
			setState(634); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(633);
				catchClause();
				}
				}
				setState(636); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__45 );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class CatchClauseContext extends ParserRuleContext {
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public ParameterListContext parameterList() {
			return getRuleContext(ParameterListContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public CatchClauseContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_catchClause; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterCatchClause(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitCatchClause(this);
		}
	}

	public final CatchClauseContext catchClause() throws RecognitionException {
		CatchClauseContext _localctx = new CatchClauseContext(_ctx, getState());
		enterRule(_localctx, 92, RULE_catchClause);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(638);
			match(T__45);
			setState(643);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796597952512L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(640);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
					{
					setState(639);
					identifier();
					}
				}

				setState(642);
				parameterList();
				}
			}

			setState(645);
			block();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class WhileStatementContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public WhileStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_whileStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterWhileStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitWhileStatement(this);
		}
	}

	public final WhileStatementContext whileStatement() throws RecognitionException {
		WhileStatementContext _localctx = new WhileStatementContext(_ctx, getState());
		enterRule(_localctx, 94, RULE_whileStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(647);
			match(T__46);
			setState(648);
			match(T__22);
			setState(649);
			expression(0);
			setState(650);
			match(T__23);
			setState(651);
			statement();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class SimpleStatementContext extends ParserRuleContext {
		public VariableDeclarationStatementContext variableDeclarationStatement() {
			return getRuleContext(VariableDeclarationStatementContext.class,0);
		}
		public ExpressionStatementContext expressionStatement() {
			return getRuleContext(ExpressionStatementContext.class,0);
		}
		public SimpleStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_simpleStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterSimpleStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitSimpleStatement(this);
		}
	}

	public final SimpleStatementContext simpleStatement() throws RecognitionException {
		SimpleStatementContext _localctx = new SimpleStatementContext(_ctx, getState());
		enterRule(_localctx, 96, RULE_simpleStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(655);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,69,_ctx) ) {
			case 1:
				{
				setState(653);
				variableDeclarationStatement();
				}
				break;
			case 2:
				{
				setState(654);
				expressionStatement();
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class UncheckedStatementContext extends ParserRuleContext {
		public BlockContext block() {
			return getRuleContext(BlockContext.class,0);
		}
		public UncheckedStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_uncheckedStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterUncheckedStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitUncheckedStatement(this);
		}
	}

	public final UncheckedStatementContext uncheckedStatement() throws RecognitionException {
		UncheckedStatementContext _localctx = new UncheckedStatementContext(_ctx, getState());
		enterRule(_localctx, 98, RULE_uncheckedStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(657);
			match(T__47);
			setState(658);
			block();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ForStatementContext extends ParserRuleContext {
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public SimpleStatementContext simpleStatement() {
			return getRuleContext(SimpleStatementContext.class,0);
		}
		public ExpressionStatementContext expressionStatement() {
			return getRuleContext(ExpressionStatementContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public ForStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_forStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterForStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitForStatement(this);
		}
	}

	public final ForStatementContext forStatement() throws RecognitionException {
		ForStatementContext _localctx = new ForStatementContext(_ctx, getState());
		enterRule(_localctx, 100, RULE_forStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(660);
			match(T__26);
			setState(661);
			match(T__22);
			setState(664);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__4:
			case T__13:
			case T__22:
			case T__24:
			case T__29:
			case T__33:
			case T__35:
			case T__37:
			case T__41:
			case T__53:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case T__58:
			case T__59:
			case T__60:
			case T__62:
			case T__63:
			case T__64:
			case T__65:
			case T__66:
			case T__95:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
			case BooleanLiteral:
			case DecimalNumber:
			case HexNumber:
			case HexLiteralFragment:
			case LeaveKeyword:
			case PayableKeyword:
			case TypeKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
			case StringLiteralFragment:
				{
				setState(662);
				simpleStatement();
				}
				break;
			case T__1:
				{
				setState(663);
				match(T__1);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(668);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__4:
			case T__13:
			case T__22:
			case T__24:
			case T__33:
			case T__35:
			case T__41:
			case T__53:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case T__58:
			case T__59:
			case T__60:
			case T__62:
			case T__63:
			case T__64:
			case T__65:
			case T__66:
			case T__95:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
			case BooleanLiteral:
			case DecimalNumber:
			case HexNumber:
			case HexLiteralFragment:
			case LeaveKeyword:
			case PayableKeyword:
			case TypeKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
			case StringLiteralFragment:
				{
				setState(666);
				expressionStatement();
				}
				break;
			case T__1:
				{
				setState(667);
				match(T__1);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(671);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
				{
				setState(670);
				expression(0);
				}
			}

			setState(673);
			match(T__23);
			setState(674);
			statement();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class InlineAssemblyStatementContext extends ParserRuleContext {
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public TerminalNode StringLiteralFragment() { return getToken(SolidityParser.StringLiteralFragment, 0); }
		public InlineAssemblyStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_inlineAssemblyStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterInlineAssemblyStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitInlineAssemblyStatement(this);
		}
	}

	public final InlineAssemblyStatementContext inlineAssemblyStatement() throws RecognitionException {
		InlineAssemblyStatementContext _localctx = new InlineAssemblyStatementContext(_ctx, getState());
		enterRule(_localctx, 102, RULE_inlineAssemblyStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(676);
			match(T__48);
			setState(678);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==StringLiteralFragment) {
				{
				setState(677);
				match(StringLiteralFragment);
				}
			}

			setState(680);
			assemblyBlock();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class DoWhileStatementContext extends ParserRuleContext {
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public DoWhileStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_doWhileStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterDoWhileStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitDoWhileStatement(this);
		}
	}

	public final DoWhileStatementContext doWhileStatement() throws RecognitionException {
		DoWhileStatementContext _localctx = new DoWhileStatementContext(_ctx, getState());
		enterRule(_localctx, 104, RULE_doWhileStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(682);
			match(T__49);
			setState(683);
			statement();
			setState(684);
			match(T__46);
			setState(685);
			match(T__22);
			setState(686);
			expression(0);
			setState(687);
			match(T__23);
			setState(688);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ContinueStatementContext extends ParserRuleContext {
		public TerminalNode ContinueKeyword() { return getToken(SolidityParser.ContinueKeyword, 0); }
		public ContinueStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_continueStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterContinueStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitContinueStatement(this);
		}
	}

	public final ContinueStatementContext continueStatement() throws RecognitionException {
		ContinueStatementContext _localctx = new ContinueStatementContext(_ctx, getState());
		enterRule(_localctx, 106, RULE_continueStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(690);
			match(ContinueKeyword);
			setState(691);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BreakStatementContext extends ParserRuleContext {
		public TerminalNode BreakKeyword() { return getToken(SolidityParser.BreakKeyword, 0); }
		public BreakStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_breakStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterBreakStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitBreakStatement(this);
		}
	}

	public final BreakStatementContext breakStatement() throws RecognitionException {
		BreakStatementContext _localctx = new BreakStatementContext(_ctx, getState());
		enterRule(_localctx, 108, RULE_breakStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(693);
			match(BreakKeyword);
			setState(694);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ReturnStatementContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public ReturnStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_returnStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterReturnStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitReturnStatement(this);
		}
	}

	public final ReturnStatementContext returnStatement() throws RecognitionException {
		ReturnStatementContext _localctx = new ReturnStatementContext(_ctx, getState());
		enterRule(_localctx, 110, RULE_returnStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(696);
			match(T__50);
			setState(698);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
				{
				setState(697);
				expression(0);
				}
			}

			setState(700);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ThrowStatementContext extends ParserRuleContext {
		public ThrowStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_throwStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterThrowStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitThrowStatement(this);
		}
	}

	public final ThrowStatementContext throwStatement() throws RecognitionException {
		ThrowStatementContext _localctx = new ThrowStatementContext(_ctx, getState());
		enterRule(_localctx, 112, RULE_throwStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(702);
			match(T__51);
			setState(703);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EmitStatementContext extends ParserRuleContext {
		public FunctionCallContext functionCall() {
			return getRuleContext(FunctionCallContext.class,0);
		}
		public EmitStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_emitStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterEmitStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitEmitStatement(this);
		}
	}

	public final EmitStatementContext emitStatement() throws RecognitionException {
		EmitStatementContext _localctx = new EmitStatementContext(_ctx, getState());
		enterRule(_localctx, 114, RULE_emitStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(705);
			match(T__52);
			setState(706);
			functionCall();
			setState(707);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class RevertStatementContext extends ParserRuleContext {
		public FunctionCallContext functionCall() {
			return getRuleContext(FunctionCallContext.class,0);
		}
		public RevertStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_revertStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterRevertStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitRevertStatement(this);
		}
	}

	public final RevertStatementContext revertStatement() throws RecognitionException {
		RevertStatementContext _localctx = new RevertStatementContext(_ctx, getState());
		enterRule(_localctx, 116, RULE_revertStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(709);
			match(T__53);
			setState(710);
			functionCall();
			setState(711);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VariableDeclarationStatementContext extends ParserRuleContext {
		public IdentifierListContext identifierList() {
			return getRuleContext(IdentifierListContext.class,0);
		}
		public VariableDeclarationContext variableDeclaration() {
			return getRuleContext(VariableDeclarationContext.class,0);
		}
		public VariableDeclarationListContext variableDeclarationList() {
			return getRuleContext(VariableDeclarationListContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public VariableDeclarationStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variableDeclarationStatement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVariableDeclarationStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVariableDeclarationStatement(this);
		}
	}

	public final VariableDeclarationStatementContext variableDeclarationStatement() throws RecognitionException {
		VariableDeclarationStatementContext _localctx = new VariableDeclarationStatementContext(_ctx, getState());
		enterRule(_localctx, 118, RULE_variableDeclarationStatement);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(720);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,75,_ctx) ) {
			case 1:
				{
				setState(713);
				match(T__54);
				setState(714);
				identifierList();
				}
				break;
			case 2:
				{
				setState(715);
				variableDeclaration();
				}
				break;
			case 3:
				{
				setState(716);
				match(T__22);
				setState(717);
				variableDeclarationList();
				setState(718);
				match(T__23);
				}
				break;
			}
			setState(724);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__9) {
				{
				setState(722);
				match(T__9);
				setState(723);
				expression(0);
				}
			}

			setState(726);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VariableDeclarationListContext extends ParserRuleContext {
		public List<VariableDeclarationContext> variableDeclaration() {
			return getRuleContexts(VariableDeclarationContext.class);
		}
		public VariableDeclarationContext variableDeclaration(int i) {
			return getRuleContext(VariableDeclarationContext.class,i);
		}
		public VariableDeclarationListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variableDeclarationList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterVariableDeclarationList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitVariableDeclarationList(this);
		}
	}

	public final VariableDeclarationListContext variableDeclarationList() throws RecognitionException {
		VariableDeclarationListContext _localctx = new VariableDeclarationListContext(_ctx, getState());
		enterRule(_localctx, 120, RULE_variableDeclarationList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(729);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
				{
				setState(728);
				variableDeclaration();
				}
			}

			setState(737);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__15) {
				{
				{
				setState(731);
				match(T__15);
				setState(733);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 558451096545148928L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778429L) != 0)) {
					{
					setState(732);
					variableDeclaration();
					}
				}

				}
				}
				setState(739);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdentifierListContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public IdentifierListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_identifierList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterIdentifierList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitIdentifierList(this);
		}
	}

	public final IdentifierListContext identifierList() throws RecognitionException {
		IdentifierListContext _localctx = new IdentifierListContext(_ctx, getState());
		enterRule(_localctx, 122, RULE_identifierList);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(740);
			match(T__22);
			setState(747);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,81,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(742);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
						{
						setState(741);
						identifier();
						}
					}

					setState(744);
					match(T__15);
					}
					} 
				}
				setState(749);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,81,_ctx);
			}
			setState(751);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(750);
				identifier();
				}
			}

			setState(753);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ElementaryTypeNameContext extends ParserRuleContext {
		public TerminalNode Int() { return getToken(SolidityParser.Int, 0); }
		public TerminalNode Uint() { return getToken(SolidityParser.Uint, 0); }
		public TerminalNode Byte() { return getToken(SolidityParser.Byte, 0); }
		public TerminalNode Fixed() { return getToken(SolidityParser.Fixed, 0); }
		public TerminalNode Ufixed() { return getToken(SolidityParser.Ufixed, 0); }
		public ElementaryTypeNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_elementaryTypeName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterElementaryTypeName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitElementaryTypeName(this);
		}
	}

	public final ElementaryTypeNameContext elementaryTypeName() throws RecognitionException {
		ElementaryTypeNameContext _localctx = new ElementaryTypeNameContext(_ctx, getState());
		enterRule(_localctx, 124, RULE_elementaryTypeName);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(755);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 540432024003936256L) != 0) || ((((_la - 98)) & ~0x3f) == 0 && ((1L << (_la - 98)) & 31L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionContext extends ParserRuleContext {
		public TypeNameContext typeName() {
			return getRuleContext(TypeNameContext.class,0);
		}
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public PrimaryExpressionContext primaryExpression() {
			return getRuleContext(PrimaryExpressionContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public NameValueListContext nameValueList() {
			return getRuleContext(NameValueListContext.class,0);
		}
		public FunctionCallArgumentsContext functionCallArguments() {
			return getRuleContext(FunctionCallArgumentsContext.class,0);
		}
		public ExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitExpression(this);
		}
	}

	public final ExpressionContext expression() throws RecognitionException {
		return expression(0);
	}

	private ExpressionContext expression(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExpressionContext _localctx = new ExpressionContext(_ctx, _parentState);
		ExpressionContext _prevctx = _localctx;
		int _startState = 126;
		enterRecursionRule(_localctx, 126, RULE_expression, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(775);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,83,_ctx) ) {
			case 1:
				{
				setState(758);
				match(T__60);
				setState(759);
				typeName(0);
				}
				break;
			case 2:
				{
				setState(760);
				match(T__22);
				setState(761);
				expression(0);
				setState(762);
				match(T__23);
				}
				break;
			case 3:
				{
				setState(764);
				_la = _input.LA(1);
				if ( !(_la==T__58 || _la==T__59) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(765);
				expression(19);
				}
				break;
			case 4:
				{
				setState(766);
				_la = _input.LA(1);
				if ( !(_la==T__62 || _la==T__63) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(767);
				expression(18);
				}
				break;
			case 5:
				{
				setState(768);
				_la = _input.LA(1);
				if ( !(_la==T__64 || _la==T__65) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(769);
				expression(17);
				}
				break;
			case 6:
				{
				setState(770);
				match(T__66);
				setState(771);
				expression(16);
				}
				break;
			case 7:
				{
				setState(772);
				match(T__4);
				setState(773);
				expression(15);
				}
				break;
			case 8:
				{
				setState(774);
				primaryExpression();
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(851);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,87,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(849);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,86,_ctx) ) {
					case 1:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(777);
						if (!(precpred(_ctx, 14))) throw new FailedPredicateException(this, "precpred(_ctx, 14)");
						setState(778);
						match(T__67);
						setState(779);
						expression(15);
						}
						break;
					case 2:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(780);
						if (!(precpred(_ctx, 13))) throw new FailedPredicateException(this, "precpred(_ctx, 13)");
						setState(781);
						_la = _input.LA(1);
						if ( !(((((_la - 13)) & ~0x3f) == 0 && ((1L << (_la - 13)) & 216172782113783809L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(782);
						expression(14);
						}
						break;
					case 3:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(783);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(784);
						_la = _input.LA(1);
						if ( !(_la==T__62 || _la==T__63) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(785);
						expression(13);
						}
						break;
					case 4:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(786);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(787);
						_la = _input.LA(1);
						if ( !(_la==T__70 || _la==T__71) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(788);
						expression(12);
						}
						break;
					case 5:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(789);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(790);
						match(T__72);
						setState(791);
						expression(11);
						}
						break;
					case 6:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(792);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(793);
						match(T__3);
						setState(794);
						expression(10);
						}
						break;
					case 7:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(795);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(796);
						match(T__73);
						setState(797);
						expression(9);
						}
						break;
					case 8:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(798);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(799);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 960L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(800);
						expression(8);
						}
						break;
					case 9:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(801);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(802);
						_la = _input.LA(1);
						if ( !(_la==T__74 || _la==T__75) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(803);
						expression(7);
						}
						break;
					case 10:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(804);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(805);
						match(T__76);
						setState(806);
						expression(6);
						}
						break;
					case 11:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(807);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(808);
						match(T__2);
						setState(809);
						expression(5);
						}
						break;
					case 12:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(810);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(811);
						match(T__77);
						setState(812);
						expression(0);
						setState(813);
						match(T__61);
						setState(814);
						expression(4);
						}
						break;
					case 13:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(816);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(817);
						_la = _input.LA(1);
						if ( !(_la==T__9 || ((((_la - 79)) & ~0x3f) == 0 && ((1L << (_la - 79)) & 1023L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(818);
						expression(3);
						}
						break;
					case 14:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(819);
						if (!(precpred(_ctx, 27))) throw new FailedPredicateException(this, "precpred(_ctx, 27)");
						setState(820);
						_la = _input.LA(1);
						if ( !(_la==T__58 || _la==T__59) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						}
						break;
					case 15:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(821);
						if (!(precpred(_ctx, 25))) throw new FailedPredicateException(this, "precpred(_ctx, 25)");
						setState(822);
						match(T__33);
						setState(823);
						expression(0);
						setState(824);
						match(T__34);
						}
						break;
					case 16:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(826);
						if (!(precpred(_ctx, 24))) throw new FailedPredicateException(this, "precpred(_ctx, 24)");
						setState(827);
						match(T__33);
						setState(829);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
							{
							setState(828);
							expression(0);
							}
						}

						setState(831);
						match(T__61);
						setState(833);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
							{
							setState(832);
							expression(0);
							}
						}

						setState(835);
						match(T__34);
						}
						break;
					case 17:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(836);
						if (!(precpred(_ctx, 23))) throw new FailedPredicateException(this, "precpred(_ctx, 23)");
						setState(837);
						match(T__36);
						setState(838);
						identifier();
						}
						break;
					case 18:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(839);
						if (!(precpred(_ctx, 22))) throw new FailedPredicateException(this, "precpred(_ctx, 22)");
						setState(840);
						match(T__14);
						setState(841);
						nameValueList();
						setState(842);
						match(T__16);
						}
						break;
					case 19:
						{
						_localctx = new ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expression);
						setState(844);
						if (!(precpred(_ctx, 21))) throw new FailedPredicateException(this, "precpred(_ctx, 21)");
						setState(845);
						match(T__22);
						setState(846);
						functionCallArguments();
						setState(847);
						match(T__23);
						}
						break;
					}
					} 
				}
				setState(853);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,87,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PrimaryExpressionContext extends ParserRuleContext {
		public TerminalNode BooleanLiteral() { return getToken(SolidityParser.BooleanLiteral, 0); }
		public NumberLiteralContext numberLiteral() {
			return getRuleContext(NumberLiteralContext.class,0);
		}
		public HexLiteralContext hexLiteral() {
			return getRuleContext(HexLiteralContext.class,0);
		}
		public StringLiteralContext stringLiteral() {
			return getRuleContext(StringLiteralContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TerminalNode TypeKeyword() { return getToken(SolidityParser.TypeKeyword, 0); }
		public TerminalNode PayableKeyword() { return getToken(SolidityParser.PayableKeyword, 0); }
		public TupleExpressionContext tupleExpression() {
			return getRuleContext(TupleExpressionContext.class,0);
		}
		public TypeNameExpressionContext typeNameExpression() {
			return getRuleContext(TypeNameExpressionContext.class,0);
		}
		public PrimaryExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_primaryExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterPrimaryExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitPrimaryExpression(this);
		}
	}

	public final PrimaryExpressionContext primaryExpression() throws RecognitionException {
		PrimaryExpressionContext _localctx = new PrimaryExpressionContext(_ctx, getState());
		enterRule(_localctx, 128, RULE_primaryExpression);
		try {
			setState(871);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,90,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(854);
				match(BooleanLiteral);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(855);
				numberLiteral();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(856);
				hexLiteral();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(857);
				stringLiteral();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(858);
				identifier();
				setState(861);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,88,_ctx) ) {
				case 1:
					{
					setState(859);
					match(T__33);
					setState(860);
					match(T__34);
					}
					break;
				}
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(863);
				match(TypeKeyword);
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(864);
				match(PayableKeyword);
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(865);
				tupleExpression();
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(866);
				typeNameExpression();
				setState(869);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,89,_ctx) ) {
				case 1:
					{
					setState(867);
					match(T__33);
					setState(868);
					match(T__34);
					}
					break;
				}
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionListContext extends ParserRuleContext {
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public ExpressionListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expressionList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterExpressionList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitExpressionList(this);
		}
	}

	public final ExpressionListContext expressionList() throws RecognitionException {
		ExpressionListContext _localctx = new ExpressionListContext(_ctx, getState());
		enterRule(_localctx, 130, RULE_expressionList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(873);
			expression(0);
			setState(878);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__15) {
				{
				{
				setState(874);
				match(T__15);
				setState(875);
				expression(0);
				}
				}
				setState(880);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class NameValueListContext extends ParserRuleContext {
		public List<NameValueContext> nameValue() {
			return getRuleContexts(NameValueContext.class);
		}
		public NameValueContext nameValue(int i) {
			return getRuleContext(NameValueContext.class,i);
		}
		public NameValueListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_nameValueList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterNameValueList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitNameValueList(this);
		}
	}

	public final NameValueListContext nameValueList() throws RecognitionException {
		NameValueListContext _localctx = new NameValueListContext(_ctx, getState());
		enterRule(_localctx, 132, RULE_nameValueList);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(881);
			nameValue();
			setState(886);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,92,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(882);
					match(T__15);
					setState(883);
					nameValue();
					}
					} 
				}
				setState(888);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,92,_ctx);
			}
			setState(890);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__15) {
				{
				setState(889);
				match(T__15);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class NameValueContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public NameValueContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_nameValue; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterNameValue(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitNameValue(this);
		}
	}

	public final NameValueContext nameValue() throws RecognitionException {
		NameValueContext _localctx = new NameValueContext(_ctx, getState());
		enterRule(_localctx, 134, RULE_nameValue);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(892);
			identifier();
			setState(893);
			match(T__61);
			setState(894);
			expression(0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionCallArgumentsContext extends ParserRuleContext {
		public NameValueListContext nameValueList() {
			return getRuleContext(NameValueListContext.class,0);
		}
		public ExpressionListContext expressionList() {
			return getRuleContext(ExpressionListContext.class,0);
		}
		public FunctionCallArgumentsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionCallArguments; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionCallArguments(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionCallArguments(this);
		}
	}

	public final FunctionCallArgumentsContext functionCallArguments() throws RecognitionException {
		FunctionCallArgumentsContext _localctx = new FunctionCallArgumentsContext(_ctx, getState());
		enterRule(_localctx, 136, RULE_functionCallArguments);
		int _la;
		try {
			setState(904);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__14:
				enterOuterAlt(_localctx, 1);
				{
				setState(896);
				match(T__14);
				setState(898);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
					{
					setState(897);
					nameValueList();
					}
				}

				setState(900);
				match(T__16);
				}
				break;
			case T__4:
			case T__13:
			case T__22:
			case T__23:
			case T__24:
			case T__33:
			case T__35:
			case T__41:
			case T__53:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case T__58:
			case T__59:
			case T__60:
			case T__62:
			case T__63:
			case T__64:
			case T__65:
			case T__66:
			case T__95:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
			case BooleanLiteral:
			case DecimalNumber:
			case HexNumber:
			case HexLiteralFragment:
			case LeaveKeyword:
			case PayableKeyword:
			case TypeKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
			case StringLiteralFragment:
				enterOuterAlt(_localctx, 2);
				{
				setState(902);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
					{
					setState(901);
					expressionList();
					}
				}

				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FunctionCallContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public FunctionCallArgumentsContext functionCallArguments() {
			return getRuleContext(FunctionCallArgumentsContext.class,0);
		}
		public FunctionCallContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_functionCall; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterFunctionCall(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitFunctionCall(this);
		}
	}

	public final FunctionCallContext functionCall() throws RecognitionException {
		FunctionCallContext _localctx = new FunctionCallContext(_ctx, getState());
		enterRule(_localctx, 138, RULE_functionCall);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(906);
			expression(0);
			setState(907);
			match(T__22);
			setState(908);
			functionCallArguments();
			setState(909);
			match(T__23);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyBlockContext extends ParserRuleContext {
		public List<AssemblyItemContext> assemblyItem() {
			return getRuleContexts(AssemblyItemContext.class);
		}
		public AssemblyItemContext assemblyItem(int i) {
			return getRuleContext(AssemblyItemContext.class,i);
		}
		public AssemblyBlockContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyBlock; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyBlock(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyBlock(this);
		}
	}

	public final AssemblyBlockContext assemblyBlock() throws RecognitionException {
		AssemblyBlockContext _localctx = new AssemblyBlockContext(_ctx, getState());
		enterRule(_localctx, 140, RULE_assemblyBlock);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(911);
			match(T__14);
			setState(915);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 309072788537262080L) != 0) || ((((_la - 89)) & ~0x3f) == 0 && ((1L << (_la - 89)) & 1993454485645L) != 0)) {
				{
				{
				setState(912);
				assemblyItem();
				}
				}
				setState(917);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(918);
			match(T__16);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyItemContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public AssemblyExpressionContext assemblyExpression() {
			return getRuleContext(AssemblyExpressionContext.class,0);
		}
		public AssemblyLocalDefinitionContext assemblyLocalDefinition() {
			return getRuleContext(AssemblyLocalDefinitionContext.class,0);
		}
		public AssemblyAssignmentContext assemblyAssignment() {
			return getRuleContext(AssemblyAssignmentContext.class,0);
		}
		public AssemblyStackAssignmentContext assemblyStackAssignment() {
			return getRuleContext(AssemblyStackAssignmentContext.class,0);
		}
		public LabelDefinitionContext labelDefinition() {
			return getRuleContext(LabelDefinitionContext.class,0);
		}
		public AssemblySwitchContext assemblySwitch() {
			return getRuleContext(AssemblySwitchContext.class,0);
		}
		public AssemblyFunctionDefinitionContext assemblyFunctionDefinition() {
			return getRuleContext(AssemblyFunctionDefinitionContext.class,0);
		}
		public AssemblyForContext assemblyFor() {
			return getRuleContext(AssemblyForContext.class,0);
		}
		public AssemblyIfContext assemblyIf() {
			return getRuleContext(AssemblyIfContext.class,0);
		}
		public TerminalNode BreakKeyword() { return getToken(SolidityParser.BreakKeyword, 0); }
		public TerminalNode ContinueKeyword() { return getToken(SolidityParser.ContinueKeyword, 0); }
		public TerminalNode LeaveKeyword() { return getToken(SolidityParser.LeaveKeyword, 0); }
		public SubAssemblyContext subAssembly() {
			return getRuleContext(SubAssemblyContext.class,0);
		}
		public NumberLiteralContext numberLiteral() {
			return getRuleContext(NumberLiteralContext.class,0);
		}
		public StringLiteralContext stringLiteral() {
			return getRuleContext(StringLiteralContext.class,0);
		}
		public HexLiteralContext hexLiteral() {
			return getRuleContext(HexLiteralContext.class,0);
		}
		public AssemblyItemContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyItem; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyItem(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyItem(this);
		}
	}

	public final AssemblyItemContext assemblyItem() throws RecognitionException {
		AssemblyItemContext _localctx = new AssemblyItemContext(_ctx, getState());
		enterRule(_localctx, 142, RULE_assemblyItem);
		try {
			setState(938);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,98,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(920);
				identifier();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(921);
				assemblyBlock();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(922);
				assemblyExpression();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(923);
				assemblyLocalDefinition();
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(924);
				assemblyAssignment();
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(925);
				assemblyStackAssignment();
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(926);
				labelDefinition();
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(927);
				assemblySwitch();
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(928);
				assemblyFunctionDefinition();
				}
				break;
			case 10:
				enterOuterAlt(_localctx, 10);
				{
				setState(929);
				assemblyFor();
				}
				break;
			case 11:
				enterOuterAlt(_localctx, 11);
				{
				setState(930);
				assemblyIf();
				}
				break;
			case 12:
				enterOuterAlt(_localctx, 12);
				{
				setState(931);
				match(BreakKeyword);
				}
				break;
			case 13:
				enterOuterAlt(_localctx, 13);
				{
				setState(932);
				match(ContinueKeyword);
				}
				break;
			case 14:
				enterOuterAlt(_localctx, 14);
				{
				setState(933);
				match(LeaveKeyword);
				}
				break;
			case 15:
				enterOuterAlt(_localctx, 15);
				{
				setState(934);
				subAssembly();
				}
				break;
			case 16:
				enterOuterAlt(_localctx, 16);
				{
				setState(935);
				numberLiteral();
				}
				break;
			case 17:
				enterOuterAlt(_localctx, 17);
				{
				setState(936);
				stringLiteral();
				}
				break;
			case 18:
				enterOuterAlt(_localctx, 18);
				{
				setState(937);
				hexLiteral();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyExpressionContext extends ParserRuleContext {
		public AssemblyCallContext assemblyCall() {
			return getRuleContext(AssemblyCallContext.class,0);
		}
		public AssemblyLiteralContext assemblyLiteral() {
			return getRuleContext(AssemblyLiteralContext.class,0);
		}
		public AssemblyMemberContext assemblyMember() {
			return getRuleContext(AssemblyMemberContext.class,0);
		}
		public AssemblyExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyExpression(this);
		}
	}

	public final AssemblyExpressionContext assemblyExpression() throws RecognitionException {
		AssemblyExpressionContext _localctx = new AssemblyExpressionContext(_ctx, getState());
		enterRule(_localctx, 144, RULE_assemblyExpression);
		try {
			setState(943);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,99,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(940);
				assemblyCall();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(941);
				assemblyLiteral();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(942);
				assemblyMember();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyMemberContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public AssemblyMemberContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyMember; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyMember(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyMember(this);
		}
	}

	public final AssemblyMemberContext assemblyMember() throws RecognitionException {
		AssemblyMemberContext _localctx = new AssemblyMemberContext(_ctx, getState());
		enterRule(_localctx, 146, RULE_assemblyMember);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(945);
			identifier();
			setState(946);
			match(T__36);
			setState(947);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyCallContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<AssemblyExpressionContext> assemblyExpression() {
			return getRuleContexts(AssemblyExpressionContext.class);
		}
		public AssemblyExpressionContext assemblyExpression(int i) {
			return getRuleContext(AssemblyExpressionContext.class,i);
		}
		public AssemblyCallContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyCall; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyCall(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyCall(this);
		}
	}

	public final AssemblyCallContext assemblyCall() throws RecognitionException {
		AssemblyCallContext _localctx = new AssemblyCallContext(_ctx, getState());
		enterRule(_localctx, 148, RULE_assemblyCall);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(953);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__50:
				{
				setState(949);
				match(T__50);
				}
				break;
			case T__35:
				{
				setState(950);
				match(T__35);
				}
				break;
			case T__57:
				{
				setState(951);
				match(T__57);
				}
				break;
			case T__13:
			case T__24:
			case T__41:
			case T__53:
			case T__95:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
				{
				setState(952);
				identifier();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(967);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,103,_ctx) ) {
			case 1:
				{
				setState(955);
				match(T__22);
				setState(957);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 308501041274437632L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15573715713L) != 0)) {
					{
					setState(956);
					assemblyExpression();
					}
				}

				setState(963);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(959);
					match(T__15);
					setState(960);
					assemblyExpression();
					}
					}
					setState(965);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(966);
				match(T__23);
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyLocalDefinitionContext extends ParserRuleContext {
		public AssemblyIdentifierOrListContext assemblyIdentifierOrList() {
			return getRuleContext(AssemblyIdentifierOrListContext.class,0);
		}
		public AssemblyExpressionContext assemblyExpression() {
			return getRuleContext(AssemblyExpressionContext.class,0);
		}
		public AssemblyLocalDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyLocalDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyLocalDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyLocalDefinition(this);
		}
	}

	public final AssemblyLocalDefinitionContext assemblyLocalDefinition() throws RecognitionException {
		AssemblyLocalDefinitionContext _localctx = new AssemblyLocalDefinitionContext(_ctx, getState());
		enterRule(_localctx, 150, RULE_assemblyLocalDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(969);
			match(T__88);
			setState(970);
			assemblyIdentifierOrList();
			setState(973);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__89) {
				{
				setState(971);
				match(T__89);
				setState(972);
				assemblyExpression();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyAssignmentContext extends ParserRuleContext {
		public AssemblyIdentifierOrListContext assemblyIdentifierOrList() {
			return getRuleContext(AssemblyIdentifierOrListContext.class,0);
		}
		public AssemblyExpressionContext assemblyExpression() {
			return getRuleContext(AssemblyExpressionContext.class,0);
		}
		public AssemblyAssignmentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyAssignment; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyAssignment(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyAssignment(this);
		}
	}

	public final AssemblyAssignmentContext assemblyAssignment() throws RecognitionException {
		AssemblyAssignmentContext _localctx = new AssemblyAssignmentContext(_ctx, getState());
		enterRule(_localctx, 152, RULE_assemblyAssignment);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(975);
			assemblyIdentifierOrList();
			setState(976);
			match(T__89);
			setState(977);
			assemblyExpression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyIdentifierOrListContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssemblyMemberContext assemblyMember() {
			return getRuleContext(AssemblyMemberContext.class,0);
		}
		public AssemblyIdentifierListContext assemblyIdentifierList() {
			return getRuleContext(AssemblyIdentifierListContext.class,0);
		}
		public AssemblyIdentifierOrListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyIdentifierOrList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyIdentifierOrList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyIdentifierOrList(this);
		}
	}

	public final AssemblyIdentifierOrListContext assemblyIdentifierOrList() throws RecognitionException {
		AssemblyIdentifierOrListContext _localctx = new AssemblyIdentifierOrListContext(_ctx, getState());
		enterRule(_localctx, 154, RULE_assemblyIdentifierOrList);
		try {
			setState(985);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,105,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(979);
				identifier();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(980);
				assemblyMember();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(981);
				match(T__22);
				setState(982);
				assemblyIdentifierList();
				setState(983);
				match(T__23);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyIdentifierListContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public AssemblyIdentifierListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyIdentifierList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyIdentifierList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyIdentifierList(this);
		}
	}

	public final AssemblyIdentifierListContext assemblyIdentifierList() throws RecognitionException {
		AssemblyIdentifierListContext _localctx = new AssemblyIdentifierListContext(_ctx, getState());
		enterRule(_localctx, 156, RULE_assemblyIdentifierList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(987);
			identifier();
			setState(992);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__15) {
				{
				{
				setState(988);
				match(T__15);
				setState(989);
				identifier();
				}
				}
				setState(994);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyStackAssignmentContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssemblyStackAssignmentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyStackAssignment; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyStackAssignment(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyStackAssignment(this);
		}
	}

	public final AssemblyStackAssignmentContext assemblyStackAssignment() throws RecognitionException {
		AssemblyStackAssignmentContext _localctx = new AssemblyStackAssignmentContext(_ctx, getState());
		enterRule(_localctx, 158, RULE_assemblyStackAssignment);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(995);
			match(T__90);
			setState(996);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LabelDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public LabelDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_labelDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterLabelDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitLabelDefinition(this);
		}
	}

	public final LabelDefinitionContext labelDefinition() throws RecognitionException {
		LabelDefinitionContext _localctx = new LabelDefinitionContext(_ctx, getState());
		enterRule(_localctx, 160, RULE_labelDefinition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(998);
			identifier();
			setState(999);
			match(T__61);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblySwitchContext extends ParserRuleContext {
		public AssemblyExpressionContext assemblyExpression() {
			return getRuleContext(AssemblyExpressionContext.class,0);
		}
		public List<AssemblyCaseContext> assemblyCase() {
			return getRuleContexts(AssemblyCaseContext.class);
		}
		public AssemblyCaseContext assemblyCase(int i) {
			return getRuleContext(AssemblyCaseContext.class,i);
		}
		public AssemblySwitchContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblySwitch; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblySwitch(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblySwitch(this);
		}
	}

	public final AssemblySwitchContext assemblySwitch() throws RecognitionException {
		AssemblySwitchContext _localctx = new AssemblySwitchContext(_ctx, getState());
		enterRule(_localctx, 162, RULE_assemblySwitch);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1001);
			match(T__91);
			setState(1002);
			assemblyExpression();
			setState(1006);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__92 || _la==T__93) {
				{
				{
				setState(1003);
				assemblyCase();
				}
				}
				setState(1008);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyCaseContext extends ParserRuleContext {
		public AssemblyLiteralContext assemblyLiteral() {
			return getRuleContext(AssemblyLiteralContext.class,0);
		}
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public AssemblyCaseContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyCase; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyCase(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyCase(this);
		}
	}

	public final AssemblyCaseContext assemblyCase() throws RecognitionException {
		AssemblyCaseContext _localctx = new AssemblyCaseContext(_ctx, getState());
		enterRule(_localctx, 164, RULE_assemblyCase);
		try {
			setState(1015);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__92:
				enterOuterAlt(_localctx, 1);
				{
				setState(1009);
				match(T__92);
				setState(1010);
				assemblyLiteral();
				setState(1011);
				assemblyBlock();
				}
				break;
			case T__93:
				enterOuterAlt(_localctx, 2);
				{
				setState(1013);
				match(T__93);
				setState(1014);
				assemblyBlock();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyFunctionDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public AssemblyIdentifierListContext assemblyIdentifierList() {
			return getRuleContext(AssemblyIdentifierListContext.class,0);
		}
		public AssemblyFunctionReturnsContext assemblyFunctionReturns() {
			return getRuleContext(AssemblyFunctionReturnsContext.class,0);
		}
		public AssemblyFunctionDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyFunctionDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyFunctionDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyFunctionDefinition(this);
		}
	}

	public final AssemblyFunctionDefinitionContext assemblyFunctionDefinition() throws RecognitionException {
		AssemblyFunctionDefinitionContext _localctx = new AssemblyFunctionDefinitionContext(_ctx, getState());
		enterRule(_localctx, 166, RULE_assemblyFunctionDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1017);
			match(T__29);
			setState(1018);
			identifier();
			setState(1019);
			match(T__22);
			setState(1021);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) {
				{
				setState(1020);
				assemblyIdentifierList();
				}
			}

			setState(1023);
			match(T__23);
			setState(1025);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__94) {
				{
				setState(1024);
				assemblyFunctionReturns();
				}
			}

			setState(1027);
			assemblyBlock();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyFunctionReturnsContext extends ParserRuleContext {
		public AssemblyIdentifierListContext assemblyIdentifierList() {
			return getRuleContext(AssemblyIdentifierListContext.class,0);
		}
		public AssemblyFunctionReturnsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyFunctionReturns; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyFunctionReturns(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyFunctionReturns(this);
		}
	}

	public final AssemblyFunctionReturnsContext assemblyFunctionReturns() throws RecognitionException {
		AssemblyFunctionReturnsContext _localctx = new AssemblyFunctionReturnsContext(_ctx, getState());
		enterRule(_localctx, 168, RULE_assemblyFunctionReturns);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(1029);
			match(T__94);
			setState(1030);
			assemblyIdentifierList();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyForContext extends ParserRuleContext {
		public List<AssemblyExpressionContext> assemblyExpression() {
			return getRuleContexts(AssemblyExpressionContext.class);
		}
		public AssemblyExpressionContext assemblyExpression(int i) {
			return getRuleContext(AssemblyExpressionContext.class,i);
		}
		public List<AssemblyBlockContext> assemblyBlock() {
			return getRuleContexts(AssemblyBlockContext.class);
		}
		public AssemblyBlockContext assemblyBlock(int i) {
			return getRuleContext(AssemblyBlockContext.class,i);
		}
		public AssemblyForContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyFor; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyFor(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyFor(this);
		}
	}

	public final AssemblyForContext assemblyFor() throws RecognitionException {
		AssemblyForContext _localctx = new AssemblyForContext(_ctx, getState());
		enterRule(_localctx, 170, RULE_assemblyFor);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1032);
			match(T__26);
			setState(1035);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__14:
				{
				setState(1033);
				assemblyBlock();
				}
				break;
			case T__13:
			case T__24:
			case T__35:
			case T__41:
			case T__50:
			case T__53:
			case T__57:
			case T__95:
			case DecimalNumber:
			case HexNumber:
			case HexLiteralFragment:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
			case StringLiteralFragment:
				{
				setState(1034);
				assemblyExpression();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(1037);
			assemblyExpression();
			setState(1040);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__14:
				{
				setState(1038);
				assemblyBlock();
				}
				break;
			case T__13:
			case T__24:
			case T__35:
			case T__41:
			case T__50:
			case T__53:
			case T__57:
			case T__95:
			case DecimalNumber:
			case HexNumber:
			case HexLiteralFragment:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
			case StringLiteralFragment:
				{
				setState(1039);
				assemblyExpression();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(1042);
			assemblyBlock();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyIfContext extends ParserRuleContext {
		public AssemblyExpressionContext assemblyExpression() {
			return getRuleContext(AssemblyExpressionContext.class,0);
		}
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public AssemblyIfContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyIf; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyIf(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyIf(this);
		}
	}

	public final AssemblyIfContext assemblyIf() throws RecognitionException {
		AssemblyIfContext _localctx = new AssemblyIfContext(_ctx, getState());
		enterRule(_localctx, 172, RULE_assemblyIf);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1044);
			match(T__42);
			setState(1045);
			assemblyExpression();
			setState(1046);
			assemblyBlock();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssemblyLiteralContext extends ParserRuleContext {
		public StringLiteralContext stringLiteral() {
			return getRuleContext(StringLiteralContext.class,0);
		}
		public TerminalNode DecimalNumber() { return getToken(SolidityParser.DecimalNumber, 0); }
		public TerminalNode HexNumber() { return getToken(SolidityParser.HexNumber, 0); }
		public HexLiteralContext hexLiteral() {
			return getRuleContext(HexLiteralContext.class,0);
		}
		public AssemblyLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assemblyLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterAssemblyLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitAssemblyLiteral(this);
		}
	}

	public final AssemblyLiteralContext assemblyLiteral() throws RecognitionException {
		AssemblyLiteralContext _localctx = new AssemblyLiteralContext(_ctx, getState());
		enterRule(_localctx, 174, RULE_assemblyLiteral);
		try {
			setState(1052);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case StringLiteralFragment:
				enterOuterAlt(_localctx, 1);
				{
				setState(1048);
				stringLiteral();
				}
				break;
			case DecimalNumber:
				enterOuterAlt(_localctx, 2);
				{
				setState(1049);
				match(DecimalNumber);
				}
				break;
			case HexNumber:
				enterOuterAlt(_localctx, 3);
				{
				setState(1050);
				match(HexNumber);
				}
				break;
			case HexLiteralFragment:
				enterOuterAlt(_localctx, 4);
				{
				setState(1051);
				hexLiteral();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class SubAssemblyContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssemblyBlockContext assemblyBlock() {
			return getRuleContext(AssemblyBlockContext.class,0);
		}
		public SubAssemblyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_subAssembly; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterSubAssembly(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitSubAssembly(this);
		}
	}

	public final SubAssemblyContext subAssembly() throws RecognitionException {
		SubAssemblyContext _localctx = new SubAssemblyContext(_ctx, getState());
		enterRule(_localctx, 176, RULE_subAssembly);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1054);
			match(T__48);
			setState(1055);
			identifier();
			setState(1056);
			assemblyBlock();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TupleExpressionContext extends ParserRuleContext {
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public TupleExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_tupleExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterTupleExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitTupleExpression(this);
		}
	}

	public final TupleExpressionContext tupleExpression() throws RecognitionException {
		TupleExpressionContext _localctx = new TupleExpressionContext(_ctx, getState());
		enterRule(_localctx, 178, RULE_tupleExpression);
		int _la;
		try {
			setState(1084);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__22:
				enterOuterAlt(_localctx, 1);
				{
				setState(1058);
				match(T__22);
				{
				setState(1060);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
					{
					setState(1059);
					expression(0);
					}
				}

				setState(1068);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(1062);
					match(T__15);
					setState(1064);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
						{
						setState(1063);
						expression(0);
						}
					}

					}
					}
					setState(1070);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
				setState(1071);
				match(T__23);
				}
				break;
			case T__33:
				enterOuterAlt(_localctx, 2);
				{
				setState(1072);
				match(T__33);
				setState(1081);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (((((_la - 5)) & ~0x3f) == 0 && ((1L << (_la - 5)) & 9078694038950117889L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 15707933693L) != 0)) {
					{
					setState(1073);
					expression(0);
					setState(1078);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__15) {
						{
						{
						setState(1074);
						match(T__15);
						setState(1075);
						expression(0);
						}
						}
						setState(1080);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(1083);
				match(T__34);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TypeNameExpressionContext extends ParserRuleContext {
		public ElementaryTypeNameContext elementaryTypeName() {
			return getRuleContext(ElementaryTypeNameContext.class,0);
		}
		public UserDefinedTypeNameContext userDefinedTypeName() {
			return getRuleContext(UserDefinedTypeNameContext.class,0);
		}
		public TypeNameExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeNameExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterTypeNameExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitTypeNameExpression(this);
		}
	}

	public final TypeNameExpressionContext typeNameExpression() throws RecognitionException {
		TypeNameExpressionContext _localctx = new TypeNameExpressionContext(_ctx, getState());
		enterRule(_localctx, 180, RULE_typeNameExpression);
		try {
			setState(1088);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__35:
			case T__54:
			case T__55:
			case T__56:
			case T__57:
			case Int:
			case Uint:
			case Byte:
			case Fixed:
			case Ufixed:
				enterOuterAlt(_localctx, 1);
				{
				setState(1086);
				elementaryTypeName();
				}
				break;
			case T__13:
			case T__24:
			case T__41:
			case T__53:
			case T__95:
			case LeaveKeyword:
			case PayableKeyword:
			case ConstructorKeyword:
			case ReceiveKeyword:
			case Identifier:
				enterOuterAlt(_localctx, 2);
				{
				setState(1087);
				userDefinedTypeName();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class NumberLiteralContext extends ParserRuleContext {
		public TerminalNode DecimalNumber() { return getToken(SolidityParser.DecimalNumber, 0); }
		public TerminalNode HexNumber() { return getToken(SolidityParser.HexNumber, 0); }
		public TerminalNode NumberUnit() { return getToken(SolidityParser.NumberUnit, 0); }
		public NumberLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_numberLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterNumberLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitNumberLiteral(this);
		}
	}

	public final NumberLiteralContext numberLiteral() throws RecognitionException {
		NumberLiteralContext _localctx = new NumberLiteralContext(_ctx, getState());
		enterRule(_localctx, 182, RULE_numberLiteral);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1090);
			_la = _input.LA(1);
			if ( !(_la==DecimalNumber || _la==HexNumber) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			setState(1092);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,121,_ctx) ) {
			case 1:
				{
				setState(1091);
				match(NumberUnit);
				}
				break;
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdentifierContext extends ParserRuleContext {
		public TerminalNode ReceiveKeyword() { return getToken(SolidityParser.ReceiveKeyword, 0); }
		public TerminalNode ConstructorKeyword() { return getToken(SolidityParser.ConstructorKeyword, 0); }
		public TerminalNode PayableKeyword() { return getToken(SolidityParser.PayableKeyword, 0); }
		public TerminalNode LeaveKeyword() { return getToken(SolidityParser.LeaveKeyword, 0); }
		public TerminalNode Identifier() { return getToken(SolidityParser.Identifier, 0); }
		public IdentifierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_identifier; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterIdentifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitIdentifier(this);
		}
	}

	public final IdentifierContext identifier() throws RecognitionException {
		IdentifierContext _localctx = new IdentifierContext(_ctx, getState());
		enterRule(_localctx, 184, RULE_identifier);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1094);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 18018796589563904L) != 0) || ((((_la - 96)) & ~0x3f) == 0 && ((1L << (_la - 96)) & 6983778305L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class HexLiteralContext extends ParserRuleContext {
		public List<TerminalNode> HexLiteralFragment() { return getTokens(SolidityParser.HexLiteralFragment); }
		public TerminalNode HexLiteralFragment(int i) {
			return getToken(SolidityParser.HexLiteralFragment, i);
		}
		public HexLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_hexLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterHexLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitHexLiteral(this);
		}
	}

	public final HexLiteralContext hexLiteral() throws RecognitionException {
		HexLiteralContext _localctx = new HexLiteralContext(_ctx, getState());
		enterRule(_localctx, 186, RULE_hexLiteral);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(1097); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(1096);
					match(HexLiteralFragment);
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(1099); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,122,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class OverrideSpecifierContext extends ParserRuleContext {
		public List<UserDefinedTypeNameContext> userDefinedTypeName() {
			return getRuleContexts(UserDefinedTypeNameContext.class);
		}
		public UserDefinedTypeNameContext userDefinedTypeName(int i) {
			return getRuleContext(UserDefinedTypeNameContext.class,i);
		}
		public OverrideSpecifierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_overrideSpecifier; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterOverrideSpecifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitOverrideSpecifier(this);
		}
	}

	public final OverrideSpecifierContext overrideSpecifier() throws RecognitionException {
		OverrideSpecifierContext _localctx = new OverrideSpecifierContext(_ctx, getState());
		enterRule(_localctx, 188, RULE_overrideSpecifier);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(1101);
			match(T__96);
			setState(1113);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__22) {
				{
				setState(1102);
				match(T__22);
				setState(1103);
				userDefinedTypeName();
				setState(1108);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__15) {
					{
					{
					setState(1104);
					match(T__15);
					setState(1105);
					userDefinedTypeName();
					}
					}
					setState(1110);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(1111);
				match(T__23);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StringLiteralContext extends ParserRuleContext {
		public List<TerminalNode> StringLiteralFragment() { return getTokens(SolidityParser.StringLiteralFragment); }
		public TerminalNode StringLiteralFragment(int i) {
			return getToken(SolidityParser.StringLiteralFragment, i);
		}
		public StringLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_stringLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).enterStringLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SolidityListener ) ((SolidityListener)listener).exitStringLiteral(this);
		}
	}

	public final StringLiteralContext stringLiteral() throws RecognitionException {
		StringLiteralContext _localctx = new StringLiteralContext(_ctx, getState());
		enterRule(_localctx, 190, RULE_stringLiteral);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(1116); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(1115);
					match(StringLiteralFragment);
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(1118); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,125,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 34:
			return typeName_sempred((TypeNameContext)_localctx, predIndex);
		case 63:
			return expression_sempred((ExpressionContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean typeName_sempred(TypeNameContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 3);
		}
		return true;
	}
	private boolean expression_sempred(ExpressionContext _localctx, int predIndex) {
		switch (predIndex) {
		case 1:
			return precpred(_ctx, 14);
		case 2:
			return precpred(_ctx, 13);
		case 3:
			return precpred(_ctx, 12);
		case 4:
			return precpred(_ctx, 11);
		case 5:
			return precpred(_ctx, 10);
		case 6:
			return precpred(_ctx, 9);
		case 7:
			return precpred(_ctx, 8);
		case 8:
			return precpred(_ctx, 7);
		case 9:
			return precpred(_ctx, 6);
		case 10:
			return precpred(_ctx, 5);
		case 11:
			return precpred(_ctx, 4);
		case 12:
			return precpred(_ctx, 3);
		case 13:
			return precpred(_ctx, 2);
		case 14:
			return precpred(_ctx, 27);
		case 15:
			return precpred(_ctx, 25);
		case 16:
			return precpred(_ctx, 24);
		case 17:
			return precpred(_ctx, 23);
		case 18:
			return precpred(_ctx, 22);
		case 19:
			return precpred(_ctx, 21);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001\u0085\u0461\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001"+
		"\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004"+
		"\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007"+
		"\u0002\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b"+
		"\u0002\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007"+
		"\u000f\u0002\u0010\u0007\u0010\u0002\u0011\u0007\u0011\u0002\u0012\u0007"+
		"\u0012\u0002\u0013\u0007\u0013\u0002\u0014\u0007\u0014\u0002\u0015\u0007"+
		"\u0015\u0002\u0016\u0007\u0016\u0002\u0017\u0007\u0017\u0002\u0018\u0007"+
		"\u0018\u0002\u0019\u0007\u0019\u0002\u001a\u0007\u001a\u0002\u001b\u0007"+
		"\u001b\u0002\u001c\u0007\u001c\u0002\u001d\u0007\u001d\u0002\u001e\u0007"+
		"\u001e\u0002\u001f\u0007\u001f\u0002 \u0007 \u0002!\u0007!\u0002\"\u0007"+
		"\"\u0002#\u0007#\u0002$\u0007$\u0002%\u0007%\u0002&\u0007&\u0002\'\u0007"+
		"\'\u0002(\u0007(\u0002)\u0007)\u0002*\u0007*\u0002+\u0007+\u0002,\u0007"+
		",\u0002-\u0007-\u0002.\u0007.\u0002/\u0007/\u00020\u00070\u00021\u0007"+
		"1\u00022\u00072\u00023\u00073\u00024\u00074\u00025\u00075\u00026\u0007"+
		"6\u00027\u00077\u00028\u00078\u00029\u00079\u0002:\u0007:\u0002;\u0007"+
		";\u0002<\u0007<\u0002=\u0007=\u0002>\u0007>\u0002?\u0007?\u0002@\u0007"+
		"@\u0002A\u0007A\u0002B\u0007B\u0002C\u0007C\u0002D\u0007D\u0002E\u0007"+
		"E\u0002F\u0007F\u0002G\u0007G\u0002H\u0007H\u0002I\u0007I\u0002J\u0007"+
		"J\u0002K\u0007K\u0002L\u0007L\u0002M\u0007M\u0002N\u0007N\u0002O\u0007"+
		"O\u0002P\u0007P\u0002Q\u0007Q\u0002R\u0007R\u0002S\u0007S\u0002T\u0007"+
		"T\u0002U\u0007U\u0002V\u0007V\u0002W\u0007W\u0002X\u0007X\u0002Y\u0007"+
		"Y\u0002Z\u0007Z\u0002[\u0007[\u0002\\\u0007\\\u0002]\u0007]\u0002^\u0007"+
		"^\u0002_\u0007_\u0001\u0000\u0001\u0000\u0001\u0000\u0001\u0000\u0001"+
		"\u0000\u0001\u0000\u0001\u0000\u0001\u0000\u0005\u0000\u00c9\b\u0000\n"+
		"\u0000\f\u0000\u00cc\t\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0002\u0001\u0002\u0001"+
		"\u0003\u0001\u0003\u0003\u0003\u00d9\b\u0003\u0001\u0004\u0001\u0004\u0003"+
		"\u0004\u00dd\b\u0004\u0001\u0004\u0005\u0004\u00e0\b\u0004\n\u0004\f\u0004"+
		"\u00e3\t\u0004\u0001\u0005\u0001\u0005\u0001\u0006\u0003\u0006\u00e8\b"+
		"\u0006\u0001\u0006\u0001\u0006\u0003\u0006\u00ec\b\u0006\u0001\u0006\u0003"+
		"\u0006\u00ef\b\u0006\u0001\u0007\u0001\u0007\u0001\u0007\u0003\u0007\u00f4"+
		"\b\u0007\u0001\b\u0001\b\u0001\b\u0001\b\u0003\b\u00fa\b\b\u0001\b\u0001"+
		"\b\u0001\b\u0001\b\u0001\b\u0003\b\u0101\b\b\u0001\b\u0001\b\u0003\b\u0105"+
		"\b\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001"+
		"\b\u0005\b\u0110\b\b\n\b\f\b\u0113\t\b\u0001\b\u0001\b\u0001\b\u0001\b"+
		"\u0001\b\u0003\b\u011a\b\b\u0001\t\u0001\t\u0001\n\u0003\n\u011f\b\n\u0001"+
		"\n\u0001\n\u0001\n\u0001\n\u0001\n\u0001\n\u0005\n\u0127\b\n\n\n\f\n\u012a"+
		"\t\n\u0003\n\u012c\b\n\u0001\n\u0001\n\u0005\n\u0130\b\n\n\n\f\n\u0133"+
		"\t\n\u0001\n\u0001\n\u0001\u000b\u0001\u000b\u0001\u000b\u0003\u000b\u013a"+
		"\b\u000b\u0001\u000b\u0003\u000b\u013d\b\u000b\u0001\f\u0001\f\u0001\f"+
		"\u0001\f\u0001\f\u0001\f\u0001\f\u0001\f\u0003\f\u0147\b\f\u0001\r\u0001"+
		"\r\u0001\r\u0001\r\u0001\r\u0001\r\u0001\r\u0005\r\u0150\b\r\n\r\f\r\u0153"+
		"\t\r\u0001\r\u0001\r\u0001\r\u0003\r\u0158\b\r\u0001\r\u0001\r\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0001\u0010"+
		"\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0003\u0010\u016d\b\u0010"+
		"\u0001\u0010\u0001\u0010\u0001\u0011\u0001\u0011\u0001\u0011\u0001\u0011"+
		"\u0001\u0011\u0001\u0011\u0001\u0011\u0001\u0011\u0005\u0011\u0179\b\u0011"+
		"\n\u0011\f\u0011\u017c\t\u0011\u0003\u0011\u017e\b\u0011\u0001\u0011\u0001"+
		"\u0011\u0001\u0012\u0001\u0012\u0001\u0012\u0003\u0012\u0185\b\u0012\u0001"+
		"\u0012\u0001\u0012\u0005\u0012\u0189\b\u0012\n\u0012\f\u0012\u018c\t\u0012"+
		"\u0001\u0012\u0001\u0012\u0003\u0012\u0190\b\u0012\u0001\u0013\u0001\u0013"+
		"\u0001\u0013\u0003\u0013\u0195\b\u0013\u0001\u0013\u0003\u0013\u0198\b"+
		"\u0013\u0001\u0014\u0001\u0014\u0001\u0014\u0001\u0014\u0003\u0014\u019e"+
		"\b\u0014\u0001\u0014\u0001\u0014\u0003\u0014\u01a2\b\u0014\u0001\u0015"+
		"\u0001\u0015\u0003\u0015\u01a6\b\u0015\u0001\u0015\u0001\u0015\u0001\u0015"+
		"\u0003\u0015\u01ab\b\u0015\u0001\u0016\u0001\u0016\u0001\u0016\u0001\u0017"+
		"\u0001\u0017\u0001\u0017\u0001\u0017\u0001\u0017\u0001\u0017\u0001\u0017"+
		"\u0001\u0017\u0005\u0017\u01b8\b\u0017\n\u0017\f\u0017\u01bb\t\u0017\u0001"+
		"\u0018\u0001\u0018\u0001\u0018\u0001\u0018\u0003\u0018\u01c1\b\u0018\u0001"+
		"\u0018\u0001\u0018\u0001\u0019\u0001\u0019\u0001\u001a\u0001\u001a\u0001"+
		"\u001a\u0001\u001a\u0003\u001a\u01cb\b\u001a\u0001\u001a\u0001\u001a\u0005"+
		"\u001a\u01cf\b\u001a\n\u001a\f\u001a\u01d2\t\u001a\u0001\u001a\u0001\u001a"+
		"\u0001\u001b\u0001\u001b\u0001\u001b\u0001\u001b\u0005\u001b\u01da\b\u001b"+
		"\n\u001b\f\u001b\u01dd\t\u001b\u0003\u001b\u01df\b\u001b\u0001\u001b\u0001"+
		"\u001b\u0001\u001c\u0001\u001c\u0003\u001c\u01e5\b\u001c\u0001\u001c\u0003"+
		"\u001c\u01e8\b\u001c\u0001\u001d\u0001\u001d\u0001\u001d\u0001\u001d\u0005"+
		"\u001d\u01ee\b\u001d\n\u001d\f\u001d\u01f1\t\u001d\u0003\u001d\u01f3\b"+
		"\u001d\u0001\u001d\u0001\u001d\u0001\u001e\u0001\u001e\u0003\u001e\u01f9"+
		"\b\u001e\u0001\u001e\u0003\u001e\u01fc\b\u001e\u0001\u001f\u0001\u001f"+
		"\u0001\u001f\u0001\u001f\u0005\u001f\u0202\b\u001f\n\u001f\f\u001f\u0205"+
		"\t\u001f\u0003\u001f\u0207\b\u001f\u0001\u001f\u0001\u001f\u0001 \u0001"+
		" \u0003 \u020d\b \u0001!\u0001!\u0003!\u0211\b!\u0001!\u0001!\u0001\""+
		"\u0001\"\u0001\"\u0001\"\u0001\"\u0001\"\u0001\"\u0003\"\u021c\b\"\u0001"+
		"\"\u0001\"\u0001\"\u0003\"\u0221\b\"\u0001\"\u0005\"\u0224\b\"\n\"\f\""+
		"\u0227\t\"\u0001#\u0001#\u0001#\u0005#\u022c\b#\n#\f#\u022f\t#\u0001$"+
		"\u0001$\u0003$\u0233\b$\u0001%\u0001%\u0001%\u0001%\u0001%\u0001%\u0001"+
		"%\u0001&\u0001&\u0001&\u0001&\u0001&\u0005&\u0241\b&\n&\f&\u0244\t&\u0001"+
		"&\u0001&\u0003&\u0248\b&\u0001\'\u0001\'\u0001(\u0001(\u0001)\u0001)\u0005"+
		")\u0250\b)\n)\f)\u0253\t)\u0001)\u0001)\u0001*\u0001*\u0001*\u0001*\u0001"+
		"*\u0001*\u0001*\u0001*\u0001*\u0001*\u0001*\u0001*\u0001*\u0001*\u0001"+
		"*\u0003*\u0266\b*\u0001+\u0001+\u0001+\u0001,\u0001,\u0001,\u0001,\u0001"+
		",\u0001,\u0001,\u0003,\u0272\b,\u0001-\u0001-\u0001-\u0003-\u0277\b-\u0001"+
		"-\u0001-\u0004-\u027b\b-\u000b-\f-\u027c\u0001.\u0001.\u0003.\u0281\b"+
		".\u0001.\u0003.\u0284\b.\u0001.\u0001.\u0001/\u0001/\u0001/\u0001/\u0001"+
		"/\u0001/\u00010\u00010\u00030\u0290\b0\u00011\u00011\u00011\u00012\u0001"+
		"2\u00012\u00012\u00032\u0299\b2\u00012\u00012\u00032\u029d\b2\u00012\u0003"+
		"2\u02a0\b2\u00012\u00012\u00012\u00013\u00013\u00033\u02a7\b3\u00013\u0001"+
		"3\u00014\u00014\u00014\u00014\u00014\u00014\u00014\u00014\u00015\u0001"+
		"5\u00015\u00016\u00016\u00016\u00017\u00017\u00037\u02bb\b7\u00017\u0001"+
		"7\u00018\u00018\u00018\u00019\u00019\u00019\u00019\u0001:\u0001:\u0001"+
		":\u0001:\u0001;\u0001;\u0001;\u0001;\u0001;\u0001;\u0001;\u0003;\u02d1"+
		"\b;\u0001;\u0001;\u0003;\u02d5\b;\u0001;\u0001;\u0001<\u0003<\u02da\b"+
		"<\u0001<\u0001<\u0003<\u02de\b<\u0005<\u02e0\b<\n<\f<\u02e3\t<\u0001="+
		"\u0001=\u0003=\u02e7\b=\u0001=\u0005=\u02ea\b=\n=\f=\u02ed\t=\u0001=\u0003"+
		"=\u02f0\b=\u0001=\u0001=\u0001>\u0001>\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0003?\u0308\b?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0003?\u033e\b?\u0001"+
		"?\u0001?\u0003?\u0342\b?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001"+
		"?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0001?\u0005?\u0352\b?\n?"+
		"\f?\u0355\t?\u0001@\u0001@\u0001@\u0001@\u0001@\u0001@\u0001@\u0003@\u035e"+
		"\b@\u0001@\u0001@\u0001@\u0001@\u0001@\u0001@\u0003@\u0366\b@\u0003@\u0368"+
		"\b@\u0001A\u0001A\u0001A\u0005A\u036d\bA\nA\fA\u0370\tA\u0001B\u0001B"+
		"\u0001B\u0005B\u0375\bB\nB\fB\u0378\tB\u0001B\u0003B\u037b\bB\u0001C\u0001"+
		"C\u0001C\u0001C\u0001D\u0001D\u0003D\u0383\bD\u0001D\u0001D\u0003D\u0387"+
		"\bD\u0003D\u0389\bD\u0001E\u0001E\u0001E\u0001E\u0001E\u0001F\u0001F\u0005"+
		"F\u0392\bF\nF\fF\u0395\tF\u0001F\u0001F\u0001G\u0001G\u0001G\u0001G\u0001"+
		"G\u0001G\u0001G\u0001G\u0001G\u0001G\u0001G\u0001G\u0001G\u0001G\u0001"+
		"G\u0001G\u0001G\u0001G\u0003G\u03ab\bG\u0001H\u0001H\u0001H\u0003H\u03b0"+
		"\bH\u0001I\u0001I\u0001I\u0001I\u0001J\u0001J\u0001J\u0001J\u0003J\u03ba"+
		"\bJ\u0001J\u0001J\u0003J\u03be\bJ\u0001J\u0001J\u0005J\u03c2\bJ\nJ\fJ"+
		"\u03c5\tJ\u0001J\u0003J\u03c8\bJ\u0001K\u0001K\u0001K\u0001K\u0003K\u03ce"+
		"\bK\u0001L\u0001L\u0001L\u0001L\u0001M\u0001M\u0001M\u0001M\u0001M\u0001"+
		"M\u0003M\u03da\bM\u0001N\u0001N\u0001N\u0005N\u03df\bN\nN\fN\u03e2\tN"+
		"\u0001O\u0001O\u0001O\u0001P\u0001P\u0001P\u0001Q\u0001Q\u0001Q\u0005"+
		"Q\u03ed\bQ\nQ\fQ\u03f0\tQ\u0001R\u0001R\u0001R\u0001R\u0001R\u0001R\u0003"+
		"R\u03f8\bR\u0001S\u0001S\u0001S\u0001S\u0003S\u03fe\bS\u0001S\u0001S\u0003"+
		"S\u0402\bS\u0001S\u0001S\u0001T\u0001T\u0001T\u0001U\u0001U\u0001U\u0003"+
		"U\u040c\bU\u0001U\u0001U\u0001U\u0003U\u0411\bU\u0001U\u0001U\u0001V\u0001"+
		"V\u0001V\u0001V\u0001W\u0001W\u0001W\u0001W\u0003W\u041d\bW\u0001X\u0001"+
		"X\u0001X\u0001X\u0001Y\u0001Y\u0003Y\u0425\bY\u0001Y\u0001Y\u0003Y\u0429"+
		"\bY\u0005Y\u042b\bY\nY\fY\u042e\tY\u0001Y\u0001Y\u0001Y\u0001Y\u0001Y"+
		"\u0005Y\u0435\bY\nY\fY\u0438\tY\u0003Y\u043a\bY\u0001Y\u0003Y\u043d\b"+
		"Y\u0001Z\u0001Z\u0003Z\u0441\bZ\u0001[\u0001[\u0003[\u0445\b[\u0001\\"+
		"\u0001\\\u0001]\u0004]\u044a\b]\u000b]\f]\u044b\u0001^\u0001^\u0001^\u0001"+
		"^\u0001^\u0005^\u0453\b^\n^\f^\u0456\t^\u0001^\u0001^\u0003^\u045a\b^"+
		"\u0001_\u0004_\u045d\b_\u000b_\f_\u045e\u0001_\u0000\u0002D~`\u0000\u0002"+
		"\u0004\u0006\b\n\f\u000e\u0010\u0012\u0014\u0016\u0018\u001a\u001c\u001e"+
		" \"$&(*,.02468:<>@BDFHJLNPRTVXZ\\^`bdfhjlnprtvxz|~\u0080\u0082\u0084\u0086"+
		"\u0088\u008a\u008c\u008e\u0090\u0092\u0094\u0096\u0098\u009a\u009c\u009e"+
		"\u00a0\u00a2\u00a4\u00a6\u00a8\u00aa\u00ac\u00ae\u00b0\u00b2\u00b4\u00b6"+
		"\u00b8\u00ba\u00bc\u00be\u0000\u000f\u0001\u0000\u0004\n\u0001\u0000\u0013"+
		"\u0015\u0001\u0000(*\u0004\u0000oovvzz||\u0003\u0000$$7:bf\u0001\u0000"+
		";<\u0001\u0000?@\u0001\u0000AB\u0002\u0000\r\rEF\u0001\u0000GH\u0001\u0000"+
		"\u0006\t\u0001\u0000KL\u0002\u0000\n\nOX\u0001\u0000hi\t\u0000\u000e\u000e"+
		"\u0019\u0019**66``rrvv}}\u007f\u0080\u04dd\u0000\u00ca\u0001\u0000\u0000"+
		"\u0000\u0002\u00cf\u0001\u0000\u0000\u0000\u0004\u00d4\u0001\u0000\u0000"+
		"\u0000\u0006\u00d8\u0001\u0000\u0000\u0000\b\u00da\u0001\u0000\u0000\u0000"+
		"\n\u00e4\u0001\u0000\u0000\u0000\f\u00ee\u0001\u0000\u0000\u0000\u000e"+
		"\u00f0\u0001\u0000\u0000\u0000\u0010\u0119\u0001\u0000\u0000\u0000\u0012"+
		"\u011b\u0001\u0000\u0000\u0000\u0014\u011e\u0001\u0000\u0000\u0000\u0016"+
		"\u0136\u0001\u0000\u0000\u0000\u0018\u0146\u0001\u0000\u0000\u0000\u001a"+
		"\u0148\u0001\u0000\u0000\u0000\u001c\u015b\u0001\u0000\u0000\u0000\u001e"+
		"\u0162\u0001\u0000\u0000\u0000 \u0167\u0001\u0000\u0000\u0000\"\u0170"+
		"\u0001\u0000\u0000\u0000$\u0181\u0001\u0000\u0000\u0000&\u0191\u0001\u0000"+
		"\u0000\u0000(\u0199\u0001\u0000\u0000\u0000*\u01aa\u0001\u0000\u0000\u0000"+
		",\u01ac\u0001\u0000\u0000\u0000.\u01b9\u0001\u0000\u0000\u00000\u01bc"+
		"\u0001\u0000\u0000\u00002\u01c4\u0001\u0000\u0000\u00004\u01c6\u0001\u0000"+
		"\u0000\u00006\u01d5\u0001\u0000\u0000\u00008\u01e2\u0001\u0000\u0000\u0000"+
		":\u01e9\u0001\u0000\u0000\u0000<\u01f6\u0001\u0000\u0000\u0000>\u01fd"+
		"\u0001\u0000\u0000\u0000@\u020a\u0001\u0000\u0000\u0000B\u020e\u0001\u0000"+
		"\u0000\u0000D\u021b\u0001\u0000\u0000\u0000F\u0228\u0001\u0000\u0000\u0000"+
		"H\u0232\u0001\u0000\u0000\u0000J\u0234\u0001\u0000\u0000\u0000L\u023b"+
		"\u0001\u0000\u0000\u0000N\u0249\u0001\u0000\u0000\u0000P\u024b\u0001\u0000"+
		"\u0000\u0000R\u024d\u0001\u0000\u0000\u0000T\u0265\u0001\u0000\u0000\u0000"+
		"V\u0267\u0001\u0000\u0000\u0000X\u026a\u0001\u0000\u0000\u0000Z\u0273"+
		"\u0001\u0000\u0000\u0000\\\u027e\u0001\u0000\u0000\u0000^\u0287\u0001"+
		"\u0000\u0000\u0000`\u028f\u0001\u0000\u0000\u0000b\u0291\u0001\u0000\u0000"+
		"\u0000d\u0294\u0001\u0000\u0000\u0000f\u02a4\u0001\u0000\u0000\u0000h"+
		"\u02aa\u0001\u0000\u0000\u0000j\u02b2\u0001\u0000\u0000\u0000l\u02b5\u0001"+
		"\u0000\u0000\u0000n\u02b8\u0001\u0000\u0000\u0000p\u02be\u0001\u0000\u0000"+
		"\u0000r\u02c1\u0001\u0000\u0000\u0000t\u02c5\u0001\u0000\u0000\u0000v"+
		"\u02d0\u0001\u0000\u0000\u0000x\u02d9\u0001\u0000\u0000\u0000z\u02e4\u0001"+
		"\u0000\u0000\u0000|\u02f3\u0001\u0000\u0000\u0000~\u0307\u0001\u0000\u0000"+
		"\u0000\u0080\u0367\u0001\u0000\u0000\u0000\u0082\u0369\u0001\u0000\u0000"+
		"\u0000\u0084\u0371\u0001\u0000\u0000\u0000\u0086\u037c\u0001\u0000\u0000"+
		"\u0000\u0088\u0388\u0001\u0000\u0000\u0000\u008a\u038a\u0001\u0000\u0000"+
		"\u0000\u008c\u038f\u0001\u0000\u0000\u0000\u008e\u03aa\u0001\u0000\u0000"+
		"\u0000\u0090\u03af\u0001\u0000\u0000\u0000\u0092\u03b1\u0001\u0000\u0000"+
		"\u0000\u0094\u03b9\u0001\u0000\u0000\u0000\u0096\u03c9\u0001\u0000\u0000"+
		"\u0000\u0098\u03cf\u0001\u0000\u0000\u0000\u009a\u03d9\u0001\u0000\u0000"+
		"\u0000\u009c\u03db\u0001\u0000\u0000\u0000\u009e\u03e3\u0001\u0000\u0000"+
		"\u0000\u00a0\u03e6\u0001\u0000\u0000\u0000\u00a2\u03e9\u0001\u0000\u0000"+
		"\u0000\u00a4\u03f7\u0001\u0000\u0000\u0000\u00a6\u03f9\u0001\u0000\u0000"+
		"\u0000\u00a8\u0405\u0001\u0000\u0000\u0000\u00aa\u0408\u0001\u0000\u0000"+
		"\u0000\u00ac\u0414\u0001\u0000\u0000\u0000\u00ae\u041c\u0001\u0000\u0000"+
		"\u0000\u00b0\u041e\u0001\u0000\u0000\u0000\u00b2\u043c\u0001\u0000\u0000"+
		"\u0000\u00b4\u0440\u0001\u0000\u0000\u0000\u00b6\u0442\u0001\u0000\u0000"+
		"\u0000\u00b8\u0446\u0001\u0000\u0000\u0000\u00ba\u0449\u0001\u0000\u0000"+
		"\u0000\u00bc\u044d\u0001\u0000\u0000\u0000\u00be\u045c\u0001\u0000\u0000"+
		"\u0000\u00c0\u00c9\u0003\u0002\u0001\u0000\u00c1\u00c9\u0003\u0010\b\u0000"+
		"\u00c2\u00c9\u0003\u0014\n\u0000\u00c3\u00c9\u00034\u001a\u0000\u00c4"+
		"\u00c9\u0003\"\u0011\u0000\u00c5\u00c9\u0003(\u0014\u0000\u00c6\u00c9"+
		"\u0003\u001c\u000e\u0000\u00c7\u00c9\u0003\u001e\u000f\u0000\u00c8\u00c0"+
		"\u0001\u0000\u0000\u0000\u00c8\u00c1\u0001\u0000\u0000\u0000\u00c8\u00c2"+
		"\u0001\u0000\u0000\u0000\u00c8\u00c3\u0001\u0000\u0000\u0000\u00c8\u00c4"+
		"\u0001\u0000\u0000\u0000\u00c8\u00c5\u0001\u0000\u0000\u0000\u00c8\u00c6"+
		"\u0001\u0000\u0000\u0000\u00c8\u00c7\u0001\u0000\u0000\u0000\u00c9\u00cc"+
		"\u0001\u0000\u0000\u0000\u00ca\u00c8\u0001\u0000\u0000\u0000\u00ca\u00cb"+
		"\u0001\u0000\u0000\u0000\u00cb\u00cd\u0001\u0000\u0000\u0000\u00cc\u00ca"+
		"\u0001\u0000\u0000\u0000\u00cd\u00ce\u0005\u0000\u0000\u0001\u00ce\u0001"+
		"\u0001\u0000\u0000\u0000\u00cf\u00d0\u0005\u0001\u0000\u0000\u00d0\u00d1"+
		"\u0003\u0004\u0002\u0000\u00d1\u00d2\u0003\u0006\u0003\u0000\u00d2\u00d3"+
		"\u0005\u0002\u0000\u0000\u00d3\u0003\u0001\u0000\u0000\u0000\u00d4\u00d5"+
		"\u0003\u00b8\\\u0000\u00d5\u0005\u0001\u0000\u0000\u0000\u00d6\u00d9\u0003"+
		"\b\u0004\u0000\u00d7\u00d9\u0003~?\u0000\u00d8\u00d6\u0001\u0000\u0000"+
		"\u0000\u00d8\u00d7\u0001\u0000\u0000\u0000\u00d9\u0007\u0001\u0000\u0000"+
		"\u0000\u00da\u00e1\u0003\f\u0006\u0000\u00db\u00dd\u0005\u0003\u0000\u0000"+
		"\u00dc\u00db\u0001\u0000\u0000\u0000\u00dc\u00dd\u0001\u0000\u0000\u0000"+
		"\u00dd\u00de\u0001\u0000\u0000\u0000\u00de\u00e0\u0003\f\u0006\u0000\u00df"+
		"\u00dc\u0001\u0000\u0000\u0000\u00e0\u00e3\u0001\u0000\u0000\u0000\u00e1"+
		"\u00df\u0001\u0000\u0000\u0000\u00e1\u00e2\u0001\u0000\u0000\u0000\u00e2"+
		"\t\u0001\u0000\u0000\u0000\u00e3\u00e1\u0001\u0000\u0000\u0000\u00e4\u00e5"+
		"\u0007\u0000\u0000\u0000\u00e5\u000b\u0001\u0000\u0000\u0000\u00e6\u00e8"+
		"\u0003\n\u0005\u0000\u00e7\u00e6\u0001\u0000\u0000\u0000\u00e7\u00e8\u0001"+
		"\u0000\u0000\u0000\u00e8\u00e9\u0001\u0000\u0000\u0000\u00e9\u00ef\u0005"+
		"\u0082\u0000\u0000\u00ea\u00ec\u0003\n\u0005\u0000\u00eb\u00ea\u0001\u0000"+
		"\u0000\u0000\u00eb\u00ec\u0001\u0000\u0000\u0000\u00ec\u00ed\u0001\u0000"+
		"\u0000\u0000\u00ed\u00ef\u0005h\u0000\u0000\u00ee\u00e7\u0001\u0000\u0000"+
		"\u0000\u00ee\u00eb\u0001\u0000\u0000\u0000\u00ef\r\u0001\u0000\u0000\u0000"+
		"\u00f0\u00f3\u0003\u00b8\\\u0000\u00f1\u00f2\u0005\u000b\u0000\u0000\u00f2"+
		"\u00f4\u0003\u00b8\\\u0000\u00f3\u00f1\u0001\u0000\u0000\u0000\u00f3\u00f4"+
		"\u0001\u0000\u0000\u0000\u00f4\u000f\u0001\u0000\u0000\u0000\u00f5\u00f6"+
		"\u0005\f\u0000\u0000\u00f6\u00f9\u0003\u0012\t\u0000\u00f7\u00f8\u0005"+
		"\u000b\u0000\u0000\u00f8\u00fa\u0003\u00b8\\\u0000\u00f9\u00f7\u0001\u0000"+
		"\u0000\u0000\u00f9\u00fa\u0001\u0000\u0000\u0000\u00fa\u00fb\u0001\u0000"+
		"\u0000\u0000\u00fb\u00fc\u0005\u0002\u0000\u0000\u00fc\u011a\u0001\u0000"+
		"\u0000\u0000\u00fd\u0100\u0005\f\u0000\u0000\u00fe\u0101\u0005\r\u0000"+
		"\u0000\u00ff\u0101\u0003\u00b8\\\u0000\u0100\u00fe\u0001\u0000\u0000\u0000"+
		"\u0100\u00ff\u0001\u0000\u0000\u0000\u0101\u0104\u0001\u0000\u0000\u0000"+
		"\u0102\u0103\u0005\u000b\u0000\u0000\u0103\u0105\u0003\u00b8\\\u0000\u0104"+
		"\u0102\u0001\u0000\u0000\u0000\u0104\u0105\u0001\u0000\u0000\u0000\u0105"+
		"\u0106\u0001\u0000\u0000\u0000\u0106\u0107\u0005\u000e\u0000\u0000\u0107"+
		"\u0108\u0003\u0012\t\u0000\u0108\u0109\u0005\u0002\u0000\u0000\u0109\u011a"+
		"\u0001\u0000\u0000\u0000\u010a\u010b\u0005\f\u0000\u0000\u010b\u010c\u0005"+
		"\u000f\u0000\u0000\u010c\u0111\u0003\u000e\u0007\u0000\u010d\u010e\u0005"+
		"\u0010\u0000\u0000\u010e\u0110\u0003\u000e\u0007\u0000\u010f\u010d\u0001"+
		"\u0000\u0000\u0000\u0110\u0113\u0001\u0000\u0000\u0000\u0111\u010f\u0001"+
		"\u0000\u0000\u0000\u0111\u0112\u0001\u0000\u0000\u0000\u0112\u0114\u0001"+
		"\u0000\u0000\u0000\u0113\u0111\u0001\u0000\u0000\u0000\u0114\u0115\u0005"+
		"\u0011\u0000\u0000\u0115\u0116\u0005\u000e\u0000\u0000\u0116\u0117\u0003"+
		"\u0012\t\u0000\u0117\u0118\u0005\u0002\u0000\u0000\u0118\u011a\u0001\u0000"+
		"\u0000\u0000\u0119\u00f5\u0001\u0000\u0000\u0000\u0119\u00fd\u0001\u0000"+
		"\u0000\u0000\u0119\u010a\u0001\u0000\u0000\u0000\u011a\u0011\u0001\u0000"+
		"\u0000\u0000\u011b\u011c\u0005\u0081\u0000\u0000\u011c\u0013\u0001\u0000"+
		"\u0000\u0000\u011d\u011f\u0005\u0012\u0000\u0000\u011e\u011d\u0001\u0000"+
		"\u0000\u0000\u011e\u011f\u0001\u0000\u0000\u0000\u011f\u0120\u0001\u0000"+
		"\u0000\u0000\u0120\u0121\u0007\u0001\u0000\u0000\u0121\u012b\u0003\u00b8"+
		"\\\u0000\u0122\u0123\u0005\u0016\u0000\u0000\u0123\u0128\u0003\u0016\u000b"+
		"\u0000\u0124\u0125\u0005\u0010\u0000\u0000\u0125\u0127\u0003\u0016\u000b"+
		"\u0000\u0126\u0124\u0001\u0000\u0000\u0000\u0127\u012a\u0001\u0000\u0000"+
		"\u0000\u0128\u0126\u0001\u0000\u0000\u0000\u0128\u0129\u0001\u0000\u0000"+
		"\u0000\u0129\u012c\u0001\u0000\u0000\u0000\u012a\u0128\u0001\u0000\u0000"+
		"\u0000\u012b\u0122\u0001\u0000\u0000\u0000\u012b\u012c\u0001\u0000\u0000"+
		"\u0000\u012c\u012d\u0001\u0000\u0000\u0000\u012d\u0131\u0005\u000f\u0000"+
		"\u0000\u012e\u0130\u0003\u0018\f\u0000\u012f\u012e\u0001\u0000\u0000\u0000"+
		"\u0130\u0133\u0001\u0000\u0000\u0000\u0131\u012f\u0001\u0000\u0000\u0000"+
		"\u0131\u0132\u0001\u0000\u0000\u0000\u0132\u0134\u0001\u0000\u0000\u0000"+
		"\u0133\u0131\u0001\u0000\u0000\u0000\u0134\u0135\u0005\u0011\u0000\u0000"+
		"\u0135\u0015\u0001\u0000\u0000\u0000\u0136\u013c\u0003F#\u0000\u0137\u0139"+
		"\u0005\u0017\u0000\u0000\u0138\u013a\u0003\u0082A\u0000\u0139\u0138\u0001"+
		"\u0000\u0000\u0000\u0139\u013a\u0001\u0000\u0000\u0000\u013a\u013b\u0001"+
		"\u0000\u0000\u0000\u013b\u013d\u0005\u0018\u0000\u0000\u013c\u0137\u0001"+
		"\u0000\u0000\u0000\u013c\u013d\u0001\u0000\u0000\u0000\u013d\u0017\u0001"+
		"\u0000\u0000\u0000\u013e\u0147\u0003\u001a\r\u0000\u013f\u0147\u0003 "+
		"\u0010\u0000\u0140\u0147\u0003\"\u0011\u0000\u0141\u0147\u0003$\u0012"+
		"\u0000\u0142\u0147\u0003(\u0014\u0000\u0143\u0147\u00030\u0018\u0000\u0144"+
		"\u0147\u00034\u001a\u0000\u0145\u0147\u0003\u001e\u000f\u0000\u0146\u013e"+
		"\u0001\u0000\u0000\u0000\u0146\u013f\u0001\u0000\u0000\u0000\u0146\u0140"+
		"\u0001\u0000\u0000\u0000\u0146\u0141\u0001\u0000\u0000\u0000\u0146\u0142"+
		"\u0001\u0000\u0000\u0000\u0146\u0143\u0001\u0000\u0000\u0000\u0146\u0144"+
		"\u0001\u0000\u0000\u0000\u0146\u0145\u0001\u0000\u0000\u0000\u0147\u0019"+
		"\u0001\u0000\u0000\u0000\u0148\u0151\u0003D\"\u0000\u0149\u0150\u0005"+
		"x\u0000\u0000\u014a\u0150\u0005u\u0000\u0000\u014b\u0150\u0005w\u0000"+
		"\u0000\u014c\u0150\u0005o\u0000\u0000\u014d\u0150\u0005p\u0000\u0000\u014e"+
		"\u0150\u0003\u00bc^\u0000\u014f\u0149\u0001\u0000\u0000\u0000\u014f\u014a"+
		"\u0001\u0000\u0000\u0000\u014f\u014b\u0001\u0000\u0000\u0000\u014f\u014c"+
		"\u0001\u0000\u0000\u0000\u014f\u014d\u0001\u0000\u0000\u0000\u014f\u014e"+
		"\u0001\u0000\u0000\u0000\u0150\u0153\u0001\u0000\u0000\u0000\u0151\u014f"+
		"\u0001\u0000\u0000\u0000\u0151\u0152\u0001\u0000\u0000\u0000\u0152\u0154"+
		"\u0001\u0000\u0000\u0000\u0153\u0151\u0001\u0000\u0000\u0000\u0154\u0157"+
		"\u0003\u00b8\\\u0000\u0155\u0156\u0005\n\u0000\u0000\u0156\u0158\u0003"+
		"~?\u0000\u0157\u0155\u0001\u0000\u0000\u0000\u0157\u0158\u0001\u0000\u0000"+
		"\u0000\u0158\u0159\u0001\u0000\u0000\u0000\u0159\u015a\u0005\u0002\u0000"+
		"\u0000\u015a\u001b\u0001\u0000\u0000\u0000\u015b\u015c\u0003D\"\u0000"+
		"\u015c\u015d\u0005o\u0000\u0000\u015d\u015e\u0003\u00b8\\\u0000\u015e"+
		"\u015f\u0005\n\u0000\u0000\u015f\u0160\u0003~?\u0000\u0160\u0161\u0005"+
		"\u0002\u0000\u0000\u0161\u001d\u0001\u0000\u0000\u0000\u0162\u0163\u0005"+
		"\u0019\u0000\u0000\u0163\u0164\u0003\u00b8\\\u0000\u0164\u0165\u00036"+
		"\u001b\u0000\u0165\u0166\u0005\u0002\u0000\u0000\u0166\u001f\u0001\u0000"+
		"\u0000\u0000\u0167\u0168\u0005\u001a\u0000\u0000\u0168\u0169\u0003\u00b8"+
		"\\\u0000\u0169\u016c\u0005\u001b\u0000\u0000\u016a\u016d\u0005\r\u0000"+
		"\u0000\u016b\u016d\u0003D\"\u0000\u016c\u016a\u0001\u0000\u0000\u0000"+
		"\u016c\u016b\u0001\u0000\u0000\u0000\u016d\u016e\u0001\u0000\u0000\u0000"+
		"\u016e\u016f\u0005\u0002\u0000\u0000\u016f!\u0001\u0000\u0000\u0000\u0170"+
		"\u0171\u0005\u001c\u0000\u0000\u0171\u0172\u0003\u00b8\\\u0000\u0172\u017d"+
		"\u0005\u000f\u0000\u0000\u0173\u0174\u0003B!\u0000\u0174\u017a\u0005\u0002"+
		"\u0000\u0000\u0175\u0176\u0003B!\u0000\u0176\u0177\u0005\u0002\u0000\u0000"+
		"\u0177\u0179\u0001\u0000\u0000\u0000\u0178\u0175\u0001\u0000\u0000\u0000"+
		"\u0179\u017c\u0001\u0000\u0000\u0000\u017a\u0178\u0001\u0000\u0000\u0000"+
		"\u017a\u017b\u0001\u0000\u0000\u0000\u017b\u017e\u0001\u0000\u0000\u0000"+
		"\u017c\u017a\u0001\u0000\u0000\u0000\u017d\u0173\u0001\u0000\u0000\u0000"+
		"\u017d\u017e\u0001\u0000\u0000\u0000\u017e\u017f\u0001\u0000\u0000\u0000"+
		"\u017f\u0180\u0005\u0011\u0000\u0000\u0180#\u0001\u0000\u0000\u0000\u0181"+
		"\u0182\u0005\u001d\u0000\u0000\u0182\u0184\u0003\u00b8\\\u0000\u0183\u0185"+
		"\u00036\u001b\u0000\u0184\u0183\u0001\u0000\u0000\u0000\u0184\u0185\u0001"+
		"\u0000\u0000\u0000\u0185\u018a\u0001\u0000\u0000\u0000\u0186\u0189\u0005"+
		"y\u0000\u0000\u0187\u0189\u0003\u00bc^\u0000\u0188\u0186\u0001\u0000\u0000"+
		"\u0000\u0188\u0187\u0001\u0000\u0000\u0000\u0189\u018c\u0001\u0000\u0000"+
		"\u0000\u018a\u0188\u0001\u0000\u0000\u0000\u018a\u018b\u0001\u0000\u0000"+
		"\u0000\u018b\u018f\u0001\u0000\u0000\u0000\u018c\u018a\u0001\u0000\u0000"+
		"\u0000\u018d\u0190\u0005\u0002\u0000\u0000\u018e\u0190\u0003R)\u0000\u018f"+
		"\u018d\u0001\u0000\u0000\u0000\u018f\u018e\u0001\u0000\u0000\u0000\u0190"+
		"%\u0001\u0000\u0000\u0000\u0191\u0197\u0003\u00b8\\\u0000\u0192\u0194"+
		"\u0005\u0017\u0000\u0000\u0193\u0195\u0003\u0082A\u0000\u0194\u0193\u0001"+
		"\u0000\u0000\u0000\u0194\u0195\u0001\u0000\u0000\u0000\u0195\u0196\u0001"+
		"\u0000\u0000\u0000\u0196\u0198\u0005\u0018\u0000\u0000\u0197\u0192\u0001"+
		"\u0000\u0000\u0000\u0197\u0198\u0001\u0000\u0000\u0000\u0198\'\u0001\u0000"+
		"\u0000\u0000\u0199\u019a\u0003*\u0015\u0000\u019a\u019b\u00036\u001b\u0000"+
		"\u019b\u019d\u0003.\u0017\u0000\u019c\u019e\u0003,\u0016\u0000\u019d\u019c"+
		"\u0001\u0000\u0000\u0000\u019d\u019e\u0001\u0000\u0000\u0000\u019e\u01a1"+
		"\u0001\u0000\u0000\u0000\u019f\u01a2\u0005\u0002\u0000\u0000\u01a0\u01a2"+
		"\u0003R)\u0000\u01a1\u019f\u0001\u0000\u0000\u0000\u01a1\u01a0\u0001\u0000"+
		"\u0000\u0000\u01a2)\u0001\u0000\u0000\u0000\u01a3\u01a5\u0005\u001e\u0000"+
		"\u0000\u01a4\u01a6\u0003\u00b8\\\u0000\u01a5\u01a4\u0001\u0000\u0000\u0000"+
		"\u01a5\u01a6\u0001\u0000\u0000\u0000\u01a6\u01ab\u0001\u0000\u0000\u0000"+
		"\u01a7\u01ab\u0005}\u0000\u0000\u01a8\u01ab\u0005~\u0000\u0000\u01a9\u01ab"+
		"\u0005\u007f\u0000\u0000\u01aa\u01a3\u0001\u0000\u0000\u0000\u01aa\u01a7"+
		"\u0001\u0000\u0000\u0000\u01aa\u01a8\u0001\u0000\u0000\u0000\u01aa\u01a9"+
		"\u0001\u0000\u0000\u0000\u01ab+\u0001\u0000\u0000\u0000\u01ac\u01ad\u0005"+
		"\u001f\u0000\u0000\u01ad\u01ae\u00036\u001b\u0000\u01ae-\u0001\u0000\u0000"+
		"\u0000\u01af\u01b8\u0005s\u0000\u0000\u01b0\u01b8\u0005x\u0000\u0000\u01b1"+
		"\u01b8\u0005u\u0000\u0000\u01b2\u01b8\u0005w\u0000\u0000\u01b3\u01b8\u0005"+
		"y\u0000\u0000\u01b4\u01b8\u0003P(\u0000\u01b5\u01b8\u0003&\u0013\u0000"+
		"\u01b6\u01b8\u0003\u00bc^\u0000\u01b7\u01af\u0001\u0000\u0000\u0000\u01b7"+
		"\u01b0\u0001\u0000\u0000\u0000\u01b7\u01b1\u0001\u0000\u0000\u0000\u01b7"+
		"\u01b2\u0001\u0000\u0000\u0000\u01b7\u01b3\u0001\u0000\u0000\u0000\u01b7"+
		"\u01b4\u0001\u0000\u0000\u0000\u01b7\u01b5\u0001\u0000\u0000\u0000\u01b7"+
		"\u01b6\u0001\u0000\u0000\u0000\u01b8\u01bb\u0001\u0000\u0000\u0000\u01b9"+
		"\u01b7\u0001\u0000\u0000\u0000\u01b9\u01ba\u0001\u0000\u0000\u0000\u01ba"+
		"/\u0001\u0000\u0000\u0000\u01bb\u01b9\u0001\u0000\u0000\u0000\u01bc\u01bd"+
		"\u0005 \u0000\u0000\u01bd\u01be\u0003\u00b8\\\u0000\u01be\u01c0\u0003"+
		":\u001d\u0000\u01bf\u01c1\u0005m\u0000\u0000\u01c0\u01bf\u0001\u0000\u0000"+
		"\u0000\u01c0\u01c1\u0001\u0000\u0000\u0000\u01c1\u01c2\u0001\u0000\u0000"+
		"\u0000\u01c2\u01c3\u0005\u0002\u0000\u0000\u01c31\u0001\u0000\u0000\u0000"+
		"\u01c4\u01c5\u0003\u00b8\\\u0000\u01c53\u0001\u0000\u0000\u0000\u01c6"+
		"\u01c7\u0005!\u0000\u0000\u01c7\u01c8\u0003\u00b8\\\u0000\u01c8\u01ca"+
		"\u0005\u000f\u0000\u0000\u01c9\u01cb\u00032\u0019\u0000\u01ca\u01c9\u0001"+
		"\u0000\u0000\u0000\u01ca\u01cb\u0001\u0000\u0000\u0000\u01cb\u01d0\u0001"+
		"\u0000\u0000\u0000\u01cc\u01cd\u0005\u0010\u0000\u0000\u01cd\u01cf\u0003"+
		"2\u0019\u0000\u01ce\u01cc\u0001\u0000\u0000\u0000\u01cf\u01d2\u0001\u0000"+
		"\u0000\u0000\u01d0\u01ce\u0001\u0000\u0000\u0000\u01d0\u01d1\u0001\u0000"+
		"\u0000\u0000\u01d1\u01d3\u0001\u0000\u0000\u0000\u01d2\u01d0\u0001\u0000"+
		"\u0000\u0000\u01d3\u01d4\u0005\u0011\u0000\u0000\u01d45\u0001\u0000\u0000"+
		"\u0000\u01d5\u01de\u0005\u0017\u0000\u0000\u01d6\u01db\u00038\u001c\u0000"+
		"\u01d7\u01d8\u0005\u0010\u0000\u0000\u01d8\u01da\u00038\u001c\u0000\u01d9"+
		"\u01d7\u0001\u0000\u0000\u0000\u01da\u01dd\u0001\u0000\u0000\u0000\u01db"+
		"\u01d9\u0001\u0000\u0000\u0000\u01db\u01dc\u0001\u0000\u0000\u0000\u01dc"+
		"\u01df\u0001\u0000\u0000\u0000\u01dd\u01db\u0001\u0000\u0000\u0000\u01de"+
		"\u01d6\u0001\u0000\u0000\u0000\u01de\u01df\u0001\u0000\u0000\u0000\u01df"+
		"\u01e0\u0001\u0000\u0000\u0000\u01e0\u01e1\u0005\u0018\u0000\u0000\u01e1"+
		"7\u0001\u0000\u0000\u0000\u01e2\u01e4\u0003D\"\u0000\u01e3\u01e5\u0003"+
		"N\'\u0000\u01e4\u01e3\u0001\u0000\u0000\u0000\u01e4\u01e5\u0001\u0000"+
		"\u0000\u0000\u01e5\u01e7\u0001\u0000\u0000\u0000\u01e6\u01e8\u0003\u00b8"+
		"\\\u0000\u01e7\u01e6\u0001\u0000\u0000\u0000\u01e7\u01e8\u0001\u0000\u0000"+
		"\u0000\u01e89\u0001\u0000\u0000\u0000\u01e9\u01f2\u0005\u0017\u0000\u0000"+
		"\u01ea\u01ef\u0003<\u001e\u0000\u01eb\u01ec\u0005\u0010\u0000\u0000\u01ec"+
		"\u01ee\u0003<\u001e\u0000\u01ed\u01eb\u0001\u0000\u0000\u0000\u01ee\u01f1"+
		"\u0001\u0000\u0000\u0000\u01ef\u01ed\u0001\u0000\u0000\u0000\u01ef\u01f0"+
		"\u0001\u0000\u0000\u0000\u01f0\u01f3\u0001\u0000\u0000\u0000\u01f1\u01ef"+
		"\u0001\u0000\u0000\u0000\u01f2\u01ea\u0001\u0000\u0000\u0000\u01f2\u01f3"+
		"\u0001\u0000\u0000\u0000\u01f3\u01f4\u0001\u0000\u0000\u0000\u01f4\u01f5"+
		"\u0005\u0018\u0000\u0000\u01f5;\u0001\u0000\u0000\u0000\u01f6\u01f8\u0003"+
		"D\"\u0000\u01f7\u01f9\u0005t\u0000\u0000\u01f8\u01f7\u0001\u0000\u0000"+
		"\u0000\u01f8\u01f9\u0001\u0000\u0000\u0000\u01f9\u01fb\u0001\u0000\u0000"+
		"\u0000\u01fa\u01fc\u0003\u00b8\\\u0000\u01fb\u01fa\u0001\u0000\u0000\u0000"+
		"\u01fb\u01fc\u0001\u0000\u0000\u0000\u01fc=\u0001\u0000\u0000\u0000\u01fd"+
		"\u0206\u0005\u0017\u0000\u0000\u01fe\u0203\u0003@ \u0000\u01ff\u0200\u0005"+
		"\u0010\u0000\u0000\u0200\u0202\u0003@ \u0000\u0201\u01ff\u0001\u0000\u0000"+
		"\u0000\u0202\u0205\u0001\u0000\u0000\u0000\u0203\u0201\u0001\u0000\u0000"+
		"\u0000\u0203\u0204\u0001\u0000\u0000\u0000\u0204\u0207\u0001\u0000\u0000"+
		"\u0000\u0205\u0203\u0001\u0000\u0000\u0000\u0206\u01fe\u0001\u0000\u0000"+
		"\u0000\u0206\u0207\u0001\u0000\u0000\u0000\u0207\u0208\u0001\u0000\u0000"+
		"\u0000\u0208\u0209\u0005\u0018\u0000\u0000\u0209?\u0001\u0000\u0000\u0000"+
		"\u020a\u020c\u0003D\"\u0000\u020b\u020d\u0003N\'\u0000\u020c\u020b\u0001"+
		"\u0000\u0000\u0000\u020c\u020d\u0001\u0000\u0000\u0000\u020dA\u0001\u0000"+
		"\u0000\u0000\u020e\u0210\u0003D\"\u0000\u020f\u0211\u0003N\'\u0000\u0210"+
		"\u020f\u0001\u0000\u0000\u0000\u0210\u0211\u0001\u0000\u0000\u0000\u0211"+
		"\u0212\u0001\u0000\u0000\u0000\u0212\u0213\u0003\u00b8\\\u0000\u0213C"+
		"\u0001\u0000\u0000\u0000\u0214\u0215\u0006\"\uffff\uffff\u0000\u0215\u021c"+
		"\u0003|>\u0000\u0216\u021c\u0003F#\u0000\u0217\u021c\u0003J%\u0000\u0218"+
		"\u021c\u0003L&\u0000\u0219\u021a\u0005$\u0000\u0000\u021a\u021c\u0005"+
		"v\u0000\u0000\u021b\u0214\u0001\u0000\u0000\u0000\u021b\u0216\u0001\u0000"+
		"\u0000\u0000\u021b\u0217\u0001\u0000\u0000\u0000\u021b\u0218\u0001\u0000"+
		"\u0000\u0000\u021b\u0219\u0001\u0000\u0000\u0000\u021c\u0225\u0001\u0000"+
		"\u0000\u0000\u021d\u021e\n\u0003\u0000\u0000\u021e\u0220\u0005\"\u0000"+
		"\u0000\u021f\u0221\u0003~?\u0000\u0220\u021f\u0001\u0000\u0000\u0000\u0220"+
		"\u0221\u0001\u0000\u0000\u0000\u0221\u0222\u0001\u0000\u0000\u0000\u0222"+
		"\u0224\u0005#\u0000\u0000\u0223\u021d\u0001\u0000\u0000\u0000\u0224\u0227"+
		"\u0001\u0000\u0000\u0000\u0225\u0223\u0001\u0000\u0000\u0000\u0225\u0226"+
		"\u0001\u0000\u0000\u0000\u0226E\u0001\u0000\u0000\u0000\u0227\u0225\u0001"+
		"\u0000\u0000\u0000\u0228\u022d\u0003\u00b8\\\u0000\u0229\u022a\u0005%"+
		"\u0000\u0000\u022a\u022c\u0003\u00b8\\\u0000\u022b\u0229\u0001\u0000\u0000"+
		"\u0000\u022c\u022f\u0001\u0000\u0000\u0000\u022d\u022b\u0001\u0000\u0000"+
		"\u0000\u022d\u022e\u0001\u0000\u0000\u0000\u022eG\u0001\u0000\u0000\u0000"+
		"\u022f\u022d\u0001\u0000\u0000\u0000\u0230\u0233\u0003|>\u0000\u0231\u0233"+
		"\u0003F#\u0000\u0232\u0230\u0001\u0000\u0000\u0000\u0232\u0231\u0001\u0000"+
		"\u0000\u0000\u0233I\u0001\u0000\u0000\u0000\u0234\u0235\u0005&\u0000\u0000"+
		"\u0235\u0236\u0005\u0017\u0000\u0000\u0236\u0237\u0003H$\u0000\u0237\u0238"+
		"\u0005\'\u0000\u0000\u0238\u0239\u0003D\"\u0000\u0239\u023a\u0005\u0018"+
		"\u0000\u0000\u023aK\u0001\u0000\u0000\u0000\u023b\u023c\u0005\u001e\u0000"+
		"\u0000\u023c\u0242\u0003>\u001f\u0000\u023d\u0241\u0005u\u0000\u0000\u023e"+
		"\u0241\u0005s\u0000\u0000\u023f\u0241\u0003P(\u0000\u0240\u023d\u0001"+
		"\u0000\u0000\u0000\u0240\u023e\u0001\u0000\u0000\u0000\u0240\u023f\u0001"+
		"\u0000\u0000\u0000\u0241\u0244\u0001\u0000\u0000\u0000\u0242\u0240\u0001"+
		"\u0000\u0000\u0000\u0242\u0243\u0001\u0000\u0000\u0000\u0243\u0247\u0001"+
		"\u0000\u0000\u0000\u0244\u0242\u0001\u0000\u0000\u0000\u0245\u0246\u0005"+
		"\u001f\u0000\u0000\u0246\u0248\u0003>\u001f\u0000\u0247\u0245\u0001\u0000"+
		"\u0000\u0000\u0247\u0248\u0001\u0000\u0000\u0000\u0248M\u0001\u0000\u0000"+
		"\u0000\u0249\u024a\u0007\u0002\u0000\u0000\u024aO\u0001\u0000\u0000\u0000"+
		"\u024b\u024c\u0007\u0003\u0000\u0000\u024cQ\u0001\u0000\u0000\u0000\u024d"+
		"\u0251\u0005\u000f\u0000\u0000\u024e\u0250\u0003T*\u0000\u024f\u024e\u0001"+
		"\u0000\u0000\u0000\u0250\u0253\u0001\u0000\u0000\u0000\u0251\u024f\u0001"+
		"\u0000\u0000\u0000\u0251\u0252\u0001\u0000\u0000\u0000\u0252\u0254\u0001"+
		"\u0000\u0000\u0000\u0253\u0251\u0001\u0000\u0000\u0000\u0254\u0255\u0005"+
		"\u0011\u0000\u0000\u0255S\u0001\u0000\u0000\u0000\u0256\u0266\u0003X,"+
		"\u0000\u0257\u0266\u0003Z-\u0000\u0258\u0266\u0003^/\u0000\u0259\u0266"+
		"\u0003d2\u0000\u025a\u0266\u0003R)\u0000\u025b\u0266\u0003f3\u0000\u025c"+
		"\u0266\u0003h4\u0000\u025d\u0266\u0003j5\u0000\u025e\u0266\u0003l6\u0000"+
		"\u025f\u0266\u0003n7\u0000\u0260\u0266\u0003p8\u0000\u0261\u0266\u0003"+
		"r9\u0000\u0262\u0266\u0003`0\u0000\u0263\u0266\u0003b1\u0000\u0264\u0266"+
		"\u0003t:\u0000\u0265\u0256\u0001\u0000\u0000\u0000\u0265\u0257\u0001\u0000"+
		"\u0000\u0000\u0265\u0258\u0001\u0000\u0000\u0000\u0265\u0259\u0001\u0000"+
		"\u0000\u0000\u0265\u025a\u0001\u0000\u0000\u0000\u0265\u025b\u0001\u0000"+
		"\u0000\u0000\u0265\u025c\u0001\u0000\u0000\u0000\u0265\u025d\u0001\u0000"+
		"\u0000\u0000\u0265\u025e\u0001\u0000\u0000\u0000\u0265\u025f\u0001\u0000"+
		"\u0000\u0000\u0265\u0260\u0001\u0000\u0000\u0000\u0265\u0261\u0001\u0000"+
		"\u0000\u0000\u0265\u0262\u0001\u0000\u0000\u0000\u0265\u0263\u0001\u0000"+
		"\u0000\u0000\u0265\u0264\u0001\u0000\u0000\u0000\u0266U\u0001\u0000\u0000"+
		"\u0000\u0267\u0268\u0003~?\u0000\u0268\u0269\u0005\u0002\u0000\u0000\u0269"+
		"W\u0001\u0000\u0000\u0000\u026a\u026b\u0005+\u0000\u0000\u026b\u026c\u0005"+
		"\u0017\u0000\u0000\u026c\u026d\u0003~?\u0000\u026d\u026e\u0005\u0018\u0000"+
		"\u0000\u026e\u0271\u0003T*\u0000\u026f\u0270\u0005,\u0000\u0000\u0270"+
		"\u0272\u0003T*\u0000\u0271\u026f\u0001\u0000\u0000\u0000\u0271\u0272\u0001"+
		"\u0000\u0000\u0000\u0272Y\u0001\u0000\u0000\u0000\u0273\u0274\u0005-\u0000"+
		"\u0000\u0274\u0276\u0003~?\u0000\u0275\u0277\u0003,\u0016\u0000\u0276"+
		"\u0275\u0001\u0000\u0000\u0000\u0276\u0277\u0001\u0000\u0000\u0000\u0277"+
		"\u0278\u0001\u0000\u0000\u0000\u0278\u027a\u0003R)\u0000\u0279\u027b\u0003"+
		"\\.\u0000\u027a\u0279\u0001\u0000\u0000\u0000\u027b\u027c\u0001\u0000"+
		"\u0000\u0000\u027c\u027a\u0001\u0000\u0000\u0000\u027c\u027d\u0001\u0000"+
		"\u0000\u0000\u027d[\u0001\u0000\u0000\u0000\u027e\u0283\u0005.\u0000\u0000"+
		"\u027f\u0281\u0003\u00b8\\\u0000\u0280\u027f\u0001\u0000\u0000\u0000\u0280"+
		"\u0281\u0001\u0000\u0000\u0000\u0281\u0282\u0001\u0000\u0000\u0000\u0282"+
		"\u0284\u00036\u001b\u0000\u0283\u0280\u0001\u0000\u0000\u0000\u0283\u0284"+
		"\u0001\u0000\u0000\u0000\u0284\u0285\u0001\u0000\u0000\u0000\u0285\u0286"+
		"\u0003R)\u0000\u0286]\u0001\u0000\u0000\u0000\u0287\u0288\u0005/\u0000"+
		"\u0000\u0288\u0289\u0005\u0017\u0000\u0000\u0289\u028a\u0003~?\u0000\u028a"+
		"\u028b\u0005\u0018\u0000\u0000\u028b\u028c\u0003T*\u0000\u028c_\u0001"+
		"\u0000\u0000\u0000\u028d\u0290\u0003v;\u0000\u028e\u0290\u0003V+\u0000"+
		"\u028f\u028d\u0001\u0000\u0000\u0000\u028f\u028e\u0001\u0000\u0000\u0000"+
		"\u0290a\u0001\u0000\u0000\u0000\u0291\u0292\u00050\u0000\u0000\u0292\u0293"+
		"\u0003R)\u0000\u0293c\u0001\u0000\u0000\u0000\u0294\u0295\u0005\u001b"+
		"\u0000\u0000\u0295\u0298\u0005\u0017\u0000\u0000\u0296\u0299\u0003`0\u0000"+
		"\u0297\u0299\u0005\u0002\u0000\u0000\u0298\u0296\u0001\u0000\u0000\u0000"+
		"\u0298\u0297\u0001\u0000\u0000\u0000\u0299\u029c\u0001\u0000\u0000\u0000"+
		"\u029a\u029d\u0003V+\u0000\u029b\u029d\u0005\u0002\u0000\u0000\u029c\u029a"+
		"\u0001\u0000\u0000\u0000\u029c\u029b\u0001\u0000\u0000\u0000\u029d\u029f"+
		"\u0001\u0000\u0000\u0000\u029e\u02a0\u0003~?\u0000\u029f\u029e\u0001\u0000"+
		"\u0000\u0000\u029f\u02a0\u0001\u0000\u0000\u0000\u02a0\u02a1\u0001\u0000"+
		"\u0000\u0000\u02a1\u02a2\u0005\u0018\u0000\u0000\u02a2\u02a3\u0003T*\u0000"+
		"\u02a3e\u0001\u0000\u0000\u0000\u02a4\u02a6\u00051\u0000\u0000\u02a5\u02a7"+
		"\u0005\u0081\u0000\u0000\u02a6\u02a5\u0001\u0000\u0000\u0000\u02a6\u02a7"+
		"\u0001\u0000\u0000\u0000\u02a7\u02a8\u0001\u0000\u0000\u0000\u02a8\u02a9"+
		"\u0003\u008cF\u0000\u02a9g\u0001\u0000\u0000\u0000\u02aa\u02ab\u00052"+
		"\u0000\u0000\u02ab\u02ac\u0003T*\u0000\u02ac\u02ad\u0005/\u0000\u0000"+
		"\u02ad\u02ae\u0005\u0017\u0000\u0000\u02ae\u02af\u0003~?\u0000\u02af\u02b0"+
		"\u0005\u0018\u0000\u0000\u02b0\u02b1\u0005\u0002\u0000\u0000\u02b1i\u0001"+
		"\u0000\u0000\u0000\u02b2\u02b3\u0005q\u0000\u0000\u02b3\u02b4\u0005\u0002"+
		"\u0000\u0000\u02b4k\u0001\u0000\u0000\u0000\u02b5\u02b6\u0005n\u0000\u0000"+
		"\u02b6\u02b7\u0005\u0002\u0000\u0000\u02b7m\u0001\u0000\u0000\u0000\u02b8"+
		"\u02ba\u00053\u0000\u0000\u02b9\u02bb\u0003~?\u0000\u02ba\u02b9\u0001"+
		"\u0000\u0000\u0000\u02ba\u02bb\u0001\u0000\u0000\u0000\u02bb\u02bc\u0001"+
		"\u0000\u0000\u0000\u02bc\u02bd\u0005\u0002\u0000\u0000\u02bdo\u0001\u0000"+
		"\u0000\u0000\u02be\u02bf\u00054\u0000\u0000\u02bf\u02c0\u0005\u0002\u0000"+
		"\u0000\u02c0q\u0001\u0000\u0000\u0000\u02c1\u02c2\u00055\u0000\u0000\u02c2"+
		"\u02c3\u0003\u008aE\u0000\u02c3\u02c4\u0005\u0002\u0000\u0000\u02c4s\u0001"+
		"\u0000\u0000\u0000\u02c5\u02c6\u00056\u0000\u0000\u02c6\u02c7\u0003\u008a"+
		"E\u0000\u02c7\u02c8\u0005\u0002\u0000\u0000\u02c8u\u0001\u0000\u0000\u0000"+
		"\u02c9\u02ca\u00057\u0000\u0000\u02ca\u02d1\u0003z=\u0000\u02cb\u02d1"+
		"\u0003B!\u0000\u02cc\u02cd\u0005\u0017\u0000\u0000\u02cd\u02ce\u0003x"+
		"<\u0000\u02ce\u02cf\u0005\u0018\u0000\u0000\u02cf\u02d1\u0001\u0000\u0000"+
		"\u0000\u02d0\u02c9\u0001\u0000\u0000\u0000\u02d0\u02cb\u0001\u0000\u0000"+
		"\u0000\u02d0\u02cc\u0001\u0000\u0000\u0000\u02d1\u02d4\u0001\u0000\u0000"+
		"\u0000\u02d2\u02d3\u0005\n\u0000\u0000\u02d3\u02d5\u0003~?\u0000\u02d4"+
		"\u02d2\u0001\u0000\u0000\u0000\u02d4\u02d5\u0001\u0000\u0000\u0000\u02d5"+
		"\u02d6\u0001\u0000\u0000\u0000\u02d6\u02d7\u0005\u0002\u0000\u0000\u02d7"+
		"w\u0001\u0000\u0000\u0000\u02d8\u02da\u0003B!\u0000\u02d9\u02d8\u0001"+
		"\u0000\u0000\u0000\u02d9\u02da\u0001\u0000\u0000\u0000\u02da\u02e1\u0001"+
		"\u0000\u0000\u0000\u02db\u02dd\u0005\u0010\u0000\u0000\u02dc\u02de\u0003"+
		"B!\u0000\u02dd\u02dc\u0001\u0000\u0000\u0000\u02dd\u02de\u0001\u0000\u0000"+
		"\u0000\u02de\u02e0\u0001\u0000\u0000\u0000\u02df\u02db\u0001\u0000\u0000"+
		"\u0000\u02e0\u02e3\u0001\u0000\u0000\u0000\u02e1\u02df\u0001\u0000\u0000"+
		"\u0000\u02e1\u02e2\u0001\u0000\u0000\u0000\u02e2y\u0001\u0000\u0000\u0000"+
		"\u02e3\u02e1\u0001\u0000\u0000\u0000\u02e4\u02eb\u0005\u0017\u0000\u0000"+
		"\u02e5\u02e7\u0003\u00b8\\\u0000\u02e6\u02e5\u0001\u0000\u0000\u0000\u02e6"+
		"\u02e7\u0001\u0000\u0000\u0000\u02e7\u02e8\u0001\u0000\u0000\u0000\u02e8"+
		"\u02ea\u0005\u0010\u0000\u0000\u02e9\u02e6\u0001\u0000\u0000\u0000\u02ea"+
		"\u02ed\u0001\u0000\u0000\u0000\u02eb\u02e9\u0001\u0000\u0000\u0000\u02eb"+
		"\u02ec\u0001\u0000\u0000\u0000\u02ec\u02ef\u0001\u0000\u0000\u0000\u02ed"+
		"\u02eb\u0001\u0000\u0000\u0000\u02ee\u02f0\u0003\u00b8\\\u0000\u02ef\u02ee"+
		"\u0001\u0000\u0000\u0000\u02ef\u02f0\u0001\u0000\u0000\u0000\u02f0\u02f1"+
		"\u0001\u0000\u0000\u0000\u02f1\u02f2\u0005\u0018\u0000\u0000\u02f2{\u0001"+
		"\u0000\u0000\u0000\u02f3\u02f4\u0007\u0004\u0000\u0000\u02f4}\u0001\u0000"+
		"\u0000\u0000\u02f5\u02f6\u0006?\uffff\uffff\u0000\u02f6\u02f7\u0005=\u0000"+
		"\u0000\u02f7\u0308\u0003D\"\u0000\u02f8\u02f9\u0005\u0017\u0000\u0000"+
		"\u02f9\u02fa\u0003~?\u0000\u02fa\u02fb\u0005\u0018\u0000\u0000\u02fb\u0308"+
		"\u0001\u0000\u0000\u0000\u02fc\u02fd\u0007\u0005\u0000\u0000\u02fd\u0308"+
		"\u0003~?\u0013\u02fe\u02ff\u0007\u0006\u0000\u0000\u02ff\u0308\u0003~"+
		"?\u0012\u0300\u0301\u0007\u0007\u0000\u0000\u0301\u0308\u0003~?\u0011"+
		"\u0302\u0303\u0005C\u0000\u0000\u0303\u0308\u0003~?\u0010\u0304\u0305"+
		"\u0005\u0005\u0000\u0000\u0305\u0308\u0003~?\u000f\u0306\u0308\u0003\u0080"+
		"@\u0000\u0307\u02f5\u0001\u0000\u0000\u0000\u0307\u02f8\u0001\u0000\u0000"+
		"\u0000\u0307\u02fc\u0001\u0000\u0000\u0000\u0307\u02fe\u0001\u0000\u0000"+
		"\u0000\u0307\u0300\u0001\u0000\u0000\u0000\u0307\u0302\u0001\u0000\u0000"+
		"\u0000\u0307\u0304\u0001\u0000\u0000\u0000\u0307\u0306\u0001\u0000\u0000"+
		"\u0000\u0308\u0353\u0001\u0000\u0000\u0000\u0309\u030a\n\u000e\u0000\u0000"+
		"\u030a\u030b\u0005D\u0000\u0000\u030b\u0352\u0003~?\u000f\u030c\u030d"+
		"\n\r\u0000\u0000\u030d\u030e\u0007\b\u0000\u0000\u030e\u0352\u0003~?\u000e"+
		"\u030f\u0310\n\f\u0000\u0000\u0310\u0311\u0007\u0006\u0000\u0000\u0311"+
		"\u0352\u0003~?\r\u0312\u0313\n\u000b\u0000\u0000\u0313\u0314\u0007\t\u0000"+
		"\u0000\u0314\u0352\u0003~?\f\u0315\u0316\n\n\u0000\u0000\u0316\u0317\u0005"+
		"I\u0000\u0000\u0317\u0352\u0003~?\u000b\u0318\u0319\n\t\u0000\u0000\u0319"+
		"\u031a\u0005\u0004\u0000\u0000\u031a\u0352\u0003~?\n\u031b\u031c\n\b\u0000"+
		"\u0000\u031c\u031d\u0005J\u0000\u0000\u031d\u0352\u0003~?\t\u031e\u031f"+
		"\n\u0007\u0000\u0000\u031f\u0320\u0007\n\u0000\u0000\u0320\u0352\u0003"+
		"~?\b\u0321\u0322\n\u0006\u0000\u0000\u0322\u0323\u0007\u000b\u0000\u0000"+
		"\u0323\u0352\u0003~?\u0007\u0324\u0325\n\u0005\u0000\u0000\u0325\u0326"+
		"\u0005M\u0000\u0000\u0326\u0352\u0003~?\u0006\u0327\u0328\n\u0004\u0000"+
		"\u0000\u0328\u0329\u0005\u0003\u0000\u0000\u0329\u0352\u0003~?\u0005\u032a"+
		"\u032b\n\u0003\u0000\u0000\u032b\u032c\u0005N\u0000\u0000\u032c\u032d"+
		"\u0003~?\u0000\u032d\u032e\u0005>\u0000\u0000\u032e\u032f\u0003~?\u0004"+
		"\u032f\u0352\u0001\u0000\u0000\u0000\u0330\u0331\n\u0002\u0000\u0000\u0331"+
		"\u0332\u0007\f\u0000\u0000\u0332\u0352\u0003~?\u0003\u0333\u0334\n\u001b"+
		"\u0000\u0000\u0334\u0352\u0007\u0005\u0000\u0000\u0335\u0336\n\u0019\u0000"+
		"\u0000\u0336\u0337\u0005\"\u0000\u0000\u0337\u0338\u0003~?\u0000\u0338"+
		"\u0339\u0005#\u0000\u0000\u0339\u0352\u0001\u0000\u0000\u0000\u033a\u033b"+
		"\n\u0018\u0000\u0000\u033b\u033d\u0005\"\u0000\u0000\u033c\u033e\u0003"+
		"~?\u0000\u033d\u033c\u0001\u0000\u0000\u0000\u033d\u033e\u0001\u0000\u0000"+
		"\u0000\u033e\u033f\u0001\u0000\u0000\u0000\u033f\u0341\u0005>\u0000\u0000"+
		"\u0340\u0342\u0003~?\u0000\u0341\u0340\u0001\u0000\u0000\u0000\u0341\u0342"+
		"\u0001\u0000\u0000\u0000\u0342\u0343\u0001\u0000\u0000\u0000\u0343\u0352"+
		"\u0005#\u0000\u0000\u0344\u0345\n\u0017\u0000\u0000\u0345\u0346\u0005"+
		"%\u0000\u0000\u0346\u0352\u0003\u00b8\\\u0000\u0347\u0348\n\u0016\u0000"+
		"\u0000\u0348\u0349\u0005\u000f\u0000\u0000\u0349\u034a\u0003\u0084B\u0000"+
		"\u034a\u034b\u0005\u0011\u0000\u0000\u034b\u0352\u0001\u0000\u0000\u0000"+
		"\u034c\u034d\n\u0015\u0000\u0000\u034d\u034e\u0005\u0017\u0000\u0000\u034e"+
		"\u034f\u0003\u0088D\u0000\u034f\u0350\u0005\u0018\u0000\u0000\u0350\u0352"+
		"\u0001\u0000\u0000\u0000\u0351\u0309\u0001\u0000\u0000\u0000\u0351\u030c"+
		"\u0001\u0000\u0000\u0000\u0351\u030f\u0001\u0000\u0000\u0000\u0351\u0312"+
		"\u0001\u0000\u0000\u0000\u0351\u0315\u0001\u0000\u0000\u0000\u0351\u0318"+
		"\u0001\u0000\u0000\u0000\u0351\u031b\u0001\u0000\u0000\u0000\u0351\u031e"+
		"\u0001\u0000\u0000\u0000\u0351\u0321\u0001\u0000\u0000\u0000\u0351\u0324"+
		"\u0001\u0000\u0000\u0000\u0351\u0327\u0001\u0000\u0000\u0000\u0351\u032a"+
		"\u0001\u0000\u0000\u0000\u0351\u0330\u0001\u0000\u0000\u0000\u0351\u0333"+
		"\u0001\u0000\u0000\u0000\u0351\u0335\u0001\u0000\u0000\u0000\u0351\u033a"+
		"\u0001\u0000\u0000\u0000\u0351\u0344\u0001\u0000\u0000\u0000\u0351\u0347"+
		"\u0001\u0000\u0000\u0000\u0351\u034c\u0001\u0000\u0000\u0000\u0352\u0355"+
		"\u0001\u0000\u0000\u0000\u0353\u0351\u0001\u0000\u0000\u0000\u0353\u0354"+
		"\u0001\u0000\u0000\u0000\u0354\u007f\u0001\u0000\u0000\u0000\u0355\u0353"+
		"\u0001\u0000\u0000\u0000\u0356\u0368\u0005g\u0000\u0000\u0357\u0368\u0003"+
		"\u00b6[\u0000\u0358\u0368\u0003\u00ba]\u0000\u0359\u0368\u0003\u00be_"+
		"\u0000\u035a\u035d\u0003\u00b8\\\u0000\u035b\u035c\u0005\"\u0000\u0000"+
		"\u035c\u035e\u0005#\u0000\u0000\u035d\u035b\u0001\u0000\u0000\u0000\u035d"+
		"\u035e\u0001\u0000\u0000\u0000\u035e\u0368\u0001\u0000\u0000\u0000\u035f"+
		"\u0368\u0005{\u0000\u0000\u0360\u0368\u0005v\u0000\u0000\u0361\u0368\u0003"+
		"\u00b2Y\u0000\u0362\u0365\u0003\u00b4Z\u0000\u0363\u0364\u0005\"\u0000"+
		"\u0000\u0364\u0366\u0005#\u0000\u0000\u0365\u0363\u0001\u0000\u0000\u0000"+
		"\u0365\u0366\u0001\u0000\u0000\u0000\u0366\u0368\u0001\u0000\u0000\u0000"+
		"\u0367\u0356\u0001\u0000\u0000\u0000\u0367\u0357\u0001\u0000\u0000\u0000"+
		"\u0367\u0358\u0001\u0000\u0000\u0000\u0367\u0359\u0001\u0000\u0000\u0000"+
		"\u0367\u035a\u0001\u0000\u0000\u0000\u0367\u035f\u0001\u0000\u0000\u0000"+
		"\u0367\u0360\u0001\u0000\u0000\u0000\u0367\u0361\u0001\u0000\u0000\u0000"+
		"\u0367\u0362\u0001\u0000\u0000\u0000\u0368\u0081\u0001\u0000\u0000\u0000"+
		"\u0369\u036e\u0003~?\u0000\u036a\u036b\u0005\u0010\u0000\u0000\u036b\u036d"+
		"\u0003~?\u0000\u036c\u036a\u0001\u0000\u0000\u0000\u036d\u0370\u0001\u0000"+
		"\u0000\u0000\u036e\u036c\u0001\u0000\u0000\u0000\u036e\u036f\u0001\u0000"+
		"\u0000\u0000\u036f\u0083\u0001\u0000\u0000\u0000\u0370\u036e\u0001\u0000"+
		"\u0000\u0000\u0371\u0376\u0003\u0086C\u0000\u0372\u0373\u0005\u0010\u0000"+
		"\u0000\u0373\u0375\u0003\u0086C\u0000\u0374\u0372\u0001\u0000\u0000\u0000"+
		"\u0375\u0378\u0001\u0000\u0000\u0000\u0376\u0374\u0001\u0000\u0000\u0000"+
		"\u0376\u0377\u0001\u0000\u0000\u0000\u0377\u037a\u0001\u0000\u0000\u0000"+
		"\u0378\u0376\u0001\u0000\u0000\u0000\u0379\u037b\u0005\u0010\u0000\u0000"+
		"\u037a\u0379\u0001\u0000\u0000\u0000\u037a\u037b\u0001\u0000\u0000\u0000"+
		"\u037b\u0085\u0001\u0000\u0000\u0000\u037c\u037d\u0003\u00b8\\\u0000\u037d"+
		"\u037e\u0005>\u0000\u0000\u037e\u037f\u0003~?\u0000\u037f\u0087\u0001"+
		"\u0000\u0000\u0000\u0380\u0382\u0005\u000f\u0000\u0000\u0381\u0383\u0003"+
		"\u0084B\u0000\u0382\u0381\u0001\u0000\u0000\u0000\u0382\u0383\u0001\u0000"+
		"\u0000\u0000\u0383\u0384\u0001\u0000\u0000\u0000\u0384\u0389\u0005\u0011"+
		"\u0000\u0000\u0385\u0387\u0003\u0082A\u0000\u0386\u0385\u0001\u0000\u0000"+
		"\u0000\u0386\u0387\u0001\u0000\u0000\u0000\u0387\u0389\u0001\u0000\u0000"+
		"\u0000\u0388\u0380\u0001\u0000\u0000\u0000\u0388\u0386\u0001\u0000\u0000"+
		"\u0000\u0389\u0089\u0001\u0000\u0000\u0000\u038a\u038b\u0003~?\u0000\u038b"+
		"\u038c\u0005\u0017\u0000\u0000\u038c\u038d\u0003\u0088D\u0000\u038d\u038e"+
		"\u0005\u0018\u0000\u0000\u038e\u008b\u0001\u0000\u0000\u0000\u038f\u0393"+
		"\u0005\u000f\u0000\u0000\u0390\u0392\u0003\u008eG\u0000\u0391\u0390\u0001"+
		"\u0000\u0000\u0000\u0392\u0395\u0001\u0000\u0000\u0000\u0393\u0391\u0001"+
		"\u0000\u0000\u0000\u0393\u0394\u0001\u0000\u0000\u0000\u0394\u0396\u0001"+
		"\u0000\u0000\u0000\u0395\u0393\u0001\u0000\u0000\u0000\u0396\u0397\u0005"+
		"\u0011\u0000\u0000\u0397\u008d\u0001\u0000\u0000\u0000\u0398\u03ab\u0003"+
		"\u00b8\\\u0000\u0399\u03ab\u0003\u008cF\u0000\u039a\u03ab\u0003\u0090"+
		"H\u0000\u039b\u03ab\u0003\u0096K\u0000\u039c\u03ab\u0003\u0098L\u0000"+
		"\u039d\u03ab\u0003\u009eO\u0000\u039e\u03ab\u0003\u00a0P\u0000\u039f\u03ab"+
		"\u0003\u00a2Q\u0000\u03a0\u03ab\u0003\u00a6S\u0000\u03a1\u03ab\u0003\u00aa"+
		"U\u0000\u03a2\u03ab\u0003\u00acV\u0000\u03a3\u03ab\u0005n\u0000\u0000"+
		"\u03a4\u03ab\u0005q\u0000\u0000\u03a5\u03ab\u0005r\u0000\u0000\u03a6\u03ab"+
		"\u0003\u00b0X\u0000\u03a7\u03ab\u0003\u00b6[\u0000\u03a8\u03ab\u0003\u00be"+
		"_\u0000\u03a9\u03ab\u0003\u00ba]\u0000\u03aa\u0398\u0001\u0000\u0000\u0000"+
		"\u03aa\u0399\u0001\u0000\u0000\u0000\u03aa\u039a\u0001\u0000\u0000\u0000"+
		"\u03aa\u039b\u0001\u0000\u0000\u0000\u03aa\u039c\u0001\u0000\u0000\u0000"+
		"\u03aa\u039d\u0001\u0000\u0000\u0000\u03aa\u039e\u0001\u0000\u0000\u0000"+
		"\u03aa\u039f\u0001\u0000\u0000\u0000\u03aa\u03a0\u0001\u0000\u0000\u0000"+
		"\u03aa\u03a1\u0001\u0000\u0000\u0000\u03aa\u03a2\u0001\u0000\u0000\u0000"+
		"\u03aa\u03a3\u0001\u0000\u0000\u0000\u03aa\u03a4\u0001\u0000\u0000\u0000"+
		"\u03aa\u03a5\u0001\u0000\u0000\u0000\u03aa\u03a6\u0001\u0000\u0000\u0000"+
		"\u03aa\u03a7\u0001\u0000\u0000\u0000\u03aa\u03a8\u0001\u0000\u0000\u0000"+
		"\u03aa\u03a9\u0001\u0000\u0000\u0000\u03ab\u008f\u0001\u0000\u0000\u0000"+
		"\u03ac\u03b0\u0003\u0094J\u0000\u03ad\u03b0\u0003\u00aeW\u0000\u03ae\u03b0"+
		"\u0003\u0092I\u0000\u03af\u03ac\u0001\u0000\u0000\u0000\u03af\u03ad\u0001"+
		"\u0000\u0000\u0000\u03af\u03ae\u0001\u0000\u0000\u0000\u03b0\u0091\u0001"+
		"\u0000\u0000\u0000\u03b1\u03b2\u0003\u00b8\\\u0000\u03b2\u03b3\u0005%"+
		"\u0000\u0000\u03b3\u03b4\u0003\u00b8\\\u0000\u03b4\u0093\u0001\u0000\u0000"+
		"\u0000\u03b5\u03ba\u00053\u0000\u0000\u03b6\u03ba\u0005$\u0000\u0000\u03b7"+
		"\u03ba\u0005:\u0000\u0000\u03b8\u03ba\u0003\u00b8\\\u0000\u03b9\u03b5"+
		"\u0001\u0000\u0000\u0000\u03b9\u03b6\u0001\u0000\u0000\u0000\u03b9\u03b7"+
		"\u0001\u0000\u0000\u0000\u03b9\u03b8\u0001\u0000\u0000\u0000\u03ba\u03c7"+
		"\u0001\u0000\u0000\u0000\u03bb\u03bd\u0005\u0017\u0000\u0000\u03bc\u03be"+
		"\u0003\u0090H\u0000\u03bd\u03bc\u0001\u0000\u0000\u0000\u03bd\u03be\u0001"+
		"\u0000\u0000\u0000\u03be\u03c3\u0001\u0000\u0000\u0000\u03bf\u03c0\u0005"+
		"\u0010\u0000\u0000\u03c0\u03c2\u0003\u0090H\u0000\u03c1\u03bf\u0001\u0000"+
		"\u0000\u0000\u03c2\u03c5\u0001\u0000\u0000\u0000\u03c3\u03c1\u0001\u0000"+
		"\u0000\u0000\u03c3\u03c4\u0001\u0000\u0000\u0000\u03c4\u03c6\u0001\u0000"+
		"\u0000\u0000\u03c5\u03c3\u0001\u0000\u0000\u0000\u03c6\u03c8\u0005\u0018"+
		"\u0000\u0000\u03c7\u03bb\u0001\u0000\u0000\u0000\u03c7\u03c8\u0001\u0000"+
		"\u0000\u0000\u03c8\u0095\u0001\u0000\u0000\u0000\u03c9\u03ca\u0005Y\u0000"+
		"\u0000\u03ca\u03cd\u0003\u009aM\u0000\u03cb\u03cc\u0005Z\u0000\u0000\u03cc"+
		"\u03ce\u0003\u0090H\u0000\u03cd\u03cb\u0001\u0000\u0000\u0000\u03cd\u03ce"+
		"\u0001\u0000\u0000\u0000\u03ce\u0097\u0001\u0000\u0000\u0000\u03cf\u03d0"+
		"\u0003\u009aM\u0000\u03d0\u03d1\u0005Z\u0000\u0000\u03d1\u03d2\u0003\u0090"+
		"H\u0000\u03d2\u0099\u0001\u0000\u0000\u0000\u03d3\u03da\u0003\u00b8\\"+
		"\u0000\u03d4\u03da\u0003\u0092I\u0000\u03d5\u03d6\u0005\u0017\u0000\u0000"+
		"\u03d6\u03d7\u0003\u009cN\u0000\u03d7\u03d8\u0005\u0018\u0000\u0000\u03d8"+
		"\u03da\u0001\u0000\u0000\u0000\u03d9\u03d3\u0001\u0000\u0000\u0000\u03d9"+
		"\u03d4\u0001\u0000\u0000\u0000\u03d9\u03d5\u0001\u0000\u0000\u0000\u03da"+
		"\u009b\u0001\u0000\u0000\u0000\u03db\u03e0\u0003\u00b8\\\u0000\u03dc\u03dd"+
		"\u0005\u0010\u0000\u0000\u03dd\u03df\u0003\u00b8\\\u0000\u03de\u03dc\u0001"+
		"\u0000\u0000\u0000\u03df\u03e2\u0001\u0000\u0000\u0000\u03e0\u03de\u0001"+
		"\u0000\u0000\u0000\u03e0\u03e1\u0001\u0000\u0000\u0000\u03e1\u009d\u0001"+
		"\u0000\u0000\u0000\u03e2\u03e0\u0001\u0000\u0000\u0000\u03e3\u03e4\u0005"+
		"[\u0000\u0000\u03e4\u03e5\u0003\u00b8\\\u0000\u03e5\u009f\u0001\u0000"+
		"\u0000\u0000\u03e6\u03e7\u0003\u00b8\\\u0000\u03e7\u03e8\u0005>\u0000"+
		"\u0000\u03e8\u00a1\u0001\u0000\u0000\u0000\u03e9\u03ea\u0005\\\u0000\u0000"+
		"\u03ea\u03ee\u0003\u0090H\u0000\u03eb\u03ed\u0003\u00a4R\u0000\u03ec\u03eb"+
		"\u0001\u0000\u0000\u0000\u03ed\u03f0\u0001\u0000\u0000\u0000\u03ee\u03ec"+
		"\u0001\u0000\u0000\u0000\u03ee\u03ef\u0001\u0000\u0000\u0000\u03ef\u00a3"+
		"\u0001\u0000\u0000\u0000\u03f0\u03ee\u0001\u0000\u0000\u0000\u03f1\u03f2"+
		"\u0005]\u0000\u0000\u03f2\u03f3\u0003\u00aeW\u0000\u03f3\u03f4\u0003\u008c"+
		"F\u0000\u03f4\u03f8\u0001\u0000\u0000\u0000\u03f5\u03f6\u0005^\u0000\u0000"+
		"\u03f6\u03f8\u0003\u008cF\u0000\u03f7\u03f1\u0001\u0000\u0000\u0000\u03f7"+
		"\u03f5\u0001\u0000\u0000\u0000\u03f8\u00a5\u0001\u0000\u0000\u0000\u03f9"+
		"\u03fa\u0005\u001e\u0000\u0000\u03fa\u03fb\u0003\u00b8\\\u0000\u03fb\u03fd"+
		"\u0005\u0017\u0000\u0000\u03fc\u03fe\u0003\u009cN\u0000\u03fd\u03fc\u0001"+
		"\u0000\u0000\u0000\u03fd\u03fe\u0001\u0000\u0000\u0000\u03fe\u03ff\u0001"+
		"\u0000\u0000\u0000\u03ff\u0401\u0005\u0018\u0000\u0000\u0400\u0402\u0003"+
		"\u00a8T\u0000\u0401\u0400\u0001\u0000\u0000\u0000\u0401\u0402\u0001\u0000"+
		"\u0000\u0000\u0402\u0403\u0001\u0000\u0000\u0000\u0403\u0404\u0003\u008c"+
		"F\u0000\u0404\u00a7\u0001\u0000\u0000\u0000\u0405\u0406\u0005_\u0000\u0000"+
		"\u0406\u0407\u0003\u009cN\u0000\u0407\u00a9\u0001\u0000\u0000\u0000\u0408"+
		"\u040b\u0005\u001b\u0000\u0000\u0409\u040c\u0003\u008cF\u0000\u040a\u040c"+
		"\u0003\u0090H\u0000\u040b\u0409\u0001\u0000\u0000\u0000\u040b\u040a\u0001"+
		"\u0000\u0000\u0000\u040c\u040d\u0001\u0000\u0000\u0000\u040d\u0410\u0003"+
		"\u0090H\u0000\u040e\u0411\u0003\u008cF\u0000\u040f\u0411\u0003\u0090H"+
		"\u0000\u0410\u040e\u0001\u0000\u0000\u0000\u0410\u040f\u0001\u0000\u0000"+
		"\u0000\u0411\u0412\u0001\u0000\u0000\u0000\u0412\u0413\u0003\u008cF\u0000"+
		"\u0413\u00ab\u0001\u0000\u0000\u0000\u0414\u0415\u0005+\u0000\u0000\u0415"+
		"\u0416\u0003\u0090H\u0000\u0416\u0417\u0003\u008cF\u0000\u0417\u00ad\u0001"+
		"\u0000\u0000\u0000\u0418\u041d\u0003\u00be_\u0000\u0419\u041d\u0005h\u0000"+
		"\u0000\u041a\u041d\u0005i\u0000\u0000\u041b\u041d\u0003\u00ba]\u0000\u041c"+
		"\u0418\u0001\u0000\u0000\u0000\u041c\u0419\u0001\u0000\u0000\u0000\u041c"+
		"\u041a\u0001\u0000\u0000\u0000\u041c\u041b\u0001\u0000\u0000\u0000\u041d"+
		"\u00af\u0001\u0000\u0000\u0000\u041e\u041f\u00051\u0000\u0000\u041f\u0420"+
		"\u0003\u00b8\\\u0000\u0420\u0421\u0003\u008cF\u0000\u0421\u00b1\u0001"+
		"\u0000\u0000\u0000\u0422\u0424\u0005\u0017\u0000\u0000\u0423\u0425\u0003"+
		"~?\u0000\u0424\u0423\u0001\u0000\u0000\u0000\u0424\u0425\u0001\u0000\u0000"+
		"\u0000\u0425\u042c\u0001\u0000\u0000\u0000\u0426\u0428\u0005\u0010\u0000"+
		"\u0000\u0427\u0429\u0003~?\u0000\u0428\u0427\u0001\u0000\u0000\u0000\u0428"+
		"\u0429\u0001\u0000\u0000\u0000\u0429\u042b\u0001\u0000\u0000\u0000\u042a"+
		"\u0426\u0001\u0000\u0000\u0000\u042b\u042e\u0001\u0000\u0000\u0000\u042c"+
		"\u042a\u0001\u0000\u0000\u0000\u042c\u042d\u0001\u0000\u0000\u0000\u042d"+
		"\u042f\u0001\u0000\u0000\u0000\u042e\u042c\u0001\u0000\u0000\u0000\u042f"+
		"\u043d\u0005\u0018\u0000\u0000\u0430\u0439\u0005\"\u0000\u0000\u0431\u0436"+
		"\u0003~?\u0000\u0432\u0433\u0005\u0010\u0000\u0000\u0433\u0435\u0003~"+
		"?\u0000\u0434\u0432\u0001\u0000\u0000\u0000\u0435\u0438\u0001\u0000\u0000"+
		"\u0000\u0436\u0434\u0001\u0000\u0000\u0000\u0436\u0437\u0001\u0000\u0000"+
		"\u0000\u0437\u043a\u0001\u0000\u0000\u0000\u0438\u0436\u0001\u0000\u0000"+
		"\u0000\u0439\u0431\u0001\u0000\u0000\u0000\u0439\u043a\u0001\u0000\u0000"+
		"\u0000\u043a\u043b\u0001\u0000\u0000\u0000\u043b\u043d\u0005#\u0000\u0000"+
		"\u043c\u0422\u0001\u0000\u0000\u0000\u043c\u0430\u0001\u0000\u0000\u0000"+
		"\u043d\u00b3\u0001\u0000\u0000\u0000\u043e\u0441\u0003|>\u0000\u043f\u0441"+
		"\u0003F#\u0000\u0440\u043e\u0001\u0000\u0000\u0000\u0440\u043f\u0001\u0000"+
		"\u0000\u0000\u0441\u00b5\u0001\u0000\u0000\u0000\u0442\u0444\u0007\r\u0000"+
		"\u0000\u0443\u0445\u0005j\u0000\u0000\u0444\u0443\u0001\u0000\u0000\u0000"+
		"\u0444\u0445\u0001\u0000\u0000\u0000\u0445\u00b7\u0001\u0000\u0000\u0000"+
		"\u0446\u0447\u0007\u000e\u0000\u0000\u0447\u00b9\u0001\u0000\u0000\u0000"+
		"\u0448\u044a\u0005k\u0000\u0000\u0449\u0448\u0001\u0000\u0000\u0000\u044a"+
		"\u044b\u0001\u0000\u0000\u0000\u044b\u0449\u0001\u0000\u0000\u0000\u044b"+
		"\u044c\u0001\u0000\u0000\u0000\u044c\u00bb\u0001\u0000\u0000\u0000\u044d"+
		"\u0459\u0005a\u0000\u0000\u044e\u044f\u0005\u0017\u0000\u0000\u044f\u0454"+
		"\u0003F#\u0000\u0450\u0451\u0005\u0010\u0000\u0000\u0451\u0453\u0003F"+
		"#\u0000\u0452\u0450\u0001\u0000\u0000\u0000\u0453\u0456\u0001\u0000\u0000"+
		"\u0000\u0454\u0452\u0001\u0000\u0000\u0000\u0454\u0455\u0001\u0000\u0000"+
		"\u0000\u0455\u0457\u0001\u0000\u0000\u0000\u0456\u0454\u0001\u0000\u0000"+
		"\u0000\u0457\u0458\u0005\u0018\u0000\u0000\u0458\u045a\u0001\u0000\u0000"+
		"\u0000\u0459\u044e\u0001\u0000\u0000\u0000\u0459\u045a\u0001\u0000\u0000"+
		"\u0000\u045a\u00bd\u0001\u0000\u0000\u0000\u045b\u045d\u0005\u0081\u0000"+
		"\u0000\u045c\u045b\u0001\u0000\u0000\u0000\u045d\u045e\u0001\u0000\u0000"+
		"\u0000\u045e\u045c\u0001\u0000\u0000\u0000\u045e\u045f\u0001\u0000\u0000"+
		"\u0000\u045f\u00bf\u0001\u0000\u0000\u0000~\u00c8\u00ca\u00d8\u00dc\u00e1"+
		"\u00e7\u00eb\u00ee\u00f3\u00f9\u0100\u0104\u0111\u0119\u011e\u0128\u012b"+
		"\u0131\u0139\u013c\u0146\u014f\u0151\u0157\u016c\u017a\u017d\u0184\u0188"+
		"\u018a\u018f\u0194\u0197\u019d\u01a1\u01a5\u01aa\u01b7\u01b9\u01c0\u01ca"+
		"\u01d0\u01db\u01de\u01e4\u01e7\u01ef\u01f2\u01f8\u01fb\u0203\u0206\u020c"+
		"\u0210\u021b\u0220\u0225\u022d\u0232\u0240\u0242\u0247\u0251\u0265\u0271"+
		"\u0276\u027c\u0280\u0283\u028f\u0298\u029c\u029f\u02a6\u02ba\u02d0\u02d4"+
		"\u02d9\u02dd\u02e1\u02e6\u02eb\u02ef\u0307\u033d\u0341\u0351\u0353\u035d"+
		"\u0365\u0367\u036e\u0376\u037a\u0382\u0386\u0388\u0393\u03aa\u03af\u03b9"+
		"\u03bd\u03c3\u03c7\u03cd\u03d9\u03e0\u03ee\u03f7\u03fd\u0401\u040b\u0410"+
		"\u041c\u0424\u0428\u042c\u0436\u0439\u043c\u0440\u0444\u044b\u0454\u0459"+
		"\u045e";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}