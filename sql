SELECT 
    table_schema || '.' || table_name AS table_full_name,
    pg_size_pretty(pg_total_relation_size(table_schema || '.' || table_name)) AS total_size,
    pg_size_pretty(pg_relation_size(table_schema || '.' || table_name)) AS table_size,
    pg_size_pretty(pg_total_relation_size(table_schema || '.' || table_name) - pg_relation_size(table_schema || '.' || table_name)) AS index_size
FROM 
    information_schema.tables
WHERE 
    table_type = 'BASE TABLE' 
    AND table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY 
    pg_total_relation_size(table_schema || '.' || table_name) DESC;

SELECT 
    nspname AS "数据库", 
    SUM(reltuples) AS "记录数", 
    SUM(pg_total_relation_size(oid) / 1024 / 1024) AS "数据容量(MB)",
    SUM(pg_indexes_size(oid) / 1024 / 1024) AS "索引容量(MB)"
FROM 
    pg_class c
JOIN 
    pg_namespace n ON c.relnamespace = n.oid
WHERE 
    relkind = 'r'  -- 只统计普通表
GROUP BY 
    nspname
ORDER BY 
    SUM(pg_total_relation_size(oid)) DESC, 
    SUM(pg_indexes_size(oid)) DESC;



SELECT 
    table_schema || '.' || table_name AS table_full_name,
    pg_size_pretty(pg_total_relation_size(table_schema || '.' || table_name)) AS total_size,
    pg_size_pretty(pg_relation_size(table_schema || '.' || table_name)) AS table_size,
    pg_size_pretty(pg_total_relation_size(table_schema || '.' || table_name) - pg_relation_size(table_schema || '.' || table_name)) AS index_size,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = t.table_schema AND table_name = t.table_name) AS column_count,
    (SELECT reltuples::BIGINT FROM pg_class WHERE relname = t.table_name) AS row_count
FROM 
    information_schema.tables t
WHERE 
    table_type = 'BASE TABLE' 
    AND table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY 
    pg_total_relation_size(table_schema || '.' || table_name) DESC;

SELECT 
    table_schema || '.' || table_name AS table_full_name,
    pg_size_pretty(pg_total_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name))) AS total_size,
    pg_size_pretty(pg_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name))) AS table_size,
    pg_size_pretty(pg_total_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name)) - pg_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name))) AS index_size,
    (SELECT COUNT(*) 
     FROM information_schema.columns 
     WHERE table_schema = t.table_schema 
     AND table_name = t.table_name) AS column_count,
    (SELECT reltuples::BIGINT 
     FROM pg_class c
     JOIN pg_namespace n ON c.relnamespace = n.oid
     WHERE c.relname = t.table_name AND n.nspname = t.table_schema) AS row_count
FROM 
    information_schema.tables t
WHERE 
    t.table_type = 'BASE TABLE'  -- 只统计普通表
    AND t.table_schema NOT IN ('pg_catalog', 'information_schema')  -- 排除系统表
ORDER BY 
    pg_total_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name)) DESC;
