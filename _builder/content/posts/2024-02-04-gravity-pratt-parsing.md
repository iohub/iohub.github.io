---
layout: posts
title: "gravity pratt parsing"
subtitle: ""
description: "pratt parsing c语言实现"
excerpt: ""
date: 2024-02-04 12:00:00
author: "rickyang"
image: "/images/posts/2.jpg"
published: true
tags:
  - python
  - huggingface
URL: "/2024/02/04/gravity_pratt_parsing"
categories:
  - parsing
  - algorithm
  - compiler
is_recommend: true
---



## TL;DR

代码路径

```c++
compiler->ast = gravity_parser_run()
	-> parser_run()
    	-> parse_statement()
    		-> parse_expression_statement()
```

解析分发函数

```c
static gnode_t *parse_statement (gravity_parser_t *parser) {
    DEBUG_PARSER("parse_statement");

    // label_statement
    // flow_statement
    // loop_statement
    // jump_statement
    // compound_statement
    // declaration_statement
    // empty_statement
    // import_statement
    // expression_statement (default)

    DECLARE_LEXER;
    gtoken_t token = gravity_lexer_peek(lexer);
    if (token_iserror(token)) return parse_error(parser);

    if (token_islabel_statement(token)) return parse_label_statement(parser);
    else if (token_isflow_statement(token)) return parse_flow_statement(parser);
    else if (token_isloop_statement(token)) return parse_loop_statement(parser);
    else if (token_isjump_statement(token)) return parse_jump_statement(parser);
    else if (token_iscompound_statement(token)) return parse_compound_statement(parser);
    else if (token_isdeclaration_statement(token)) return parse_declaration_statement(parser);
    else if (token_isempty_statement(token)) return parse_empty_statement(parser);
    else if (token_isimport_statement(token)) return parse_import_statement(parser);
    else if (token_isspecial_statement(token)) return parse_special_statement(parser);
    else if (token_ismacro(token)) return parse_macro_statement(parser);

    return parse_expression_statement(parser); // DEFAULT
}
```

pratt parsing执行函数

```c
static gnode_t *parse_expression_statement (gravity_parser_t *parser) {
    DEBUG_PARSER("parse_expression_statement");

    gnode_t *expr = parse_expression(parser);
    parse_semicolon(parser);
    return expr;
}

static gnode_t *parse_expression (gravity_parser_t *parser) {
    DEBUG_PARSER("parse_expression");
    return parse_precedence(parser, PREC_LOWEST);
}
```





## Pratt Parsing实现

```sh
static gnode_t *parse_precedence(gravity_parser_t *parser, prec_level precedence) {
    DEBUG_PARSER("parse_precedence (level %d)", precedence);
    DECLARE_LEXER;

    // peek next and check for EOF
    gtoken_t type = gravity_lexer_peek(lexer);
    if (type == TOK_EOF) return NULL;

    // execute prefix callback (if any)
    parse_func prefix = rules[type].prefix;

    // to protect stack from excessive recursion
    if (prefix && (++parser->expr_depth > MAX_EXPRESSION_DEPTH)) {
        // consume next token to avoid infinite loops
        gravity_lexer_next(lexer);
        REPORT_ERROR(gravity_lexer_token(lexer), "Maximum expression depth reached.");
        return NULL;
    }
    gnode_t *node = (prefix) ? prefix(parser) : NULL;
    if (prefix) --parser->expr_depth;

    if (!prefix || !node) {
        // we need to consume next token because error was triggered in peek
        gravity_lexer_next(lexer);
        REPORT_ERROR(gravity_lexer_token(lexer), "Expected expression but found %s.", token_name(type));
        return NULL;
    }

    // peek next and check for EOF
    gtoken_t peek = gravity_lexer_peek(lexer);
    if (peek == TOK_EOF) return node;

    while (precedence < rules[peek].precedence) {
        gtoken_t tok = gravity_lexer_next(lexer);
        grammar_rule *rule = &rules[tok];

        // execute infix callback
        parser->current_token = tok;
        parser->current_node = node;
        node = rule->infix(parser);

        // peek next and check for EOF
        peek = gravity_lexer_peek(lexer);
        if (peek == TOK_EOF) break;
    }

    return node;
}

```

