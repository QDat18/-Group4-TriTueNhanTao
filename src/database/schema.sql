create table employees (
    employee_id text primary key,

    full_name text not null,

    department text,

    position text,

    email text,

    phone text,

    is_active boolean default true,

    created_at timestamptz default now()
);

create table face_embeddings (

    employee_id text primary key,

    full_name text,

    embedding_vector jsonb not null,

    image_count integer default 0,

    updated_at timestamptz default now(),

    constraint fk_embedding_employee
        foreign key(employee_id)
        references employees(employee_id)
);

create table attendance_logs (
    attendance_id bigint generated always as identity primary key,

    employee_id text not null,

    check_time timestamptz default now(),

    similarity float,

    camera_id text,

    status text,

    constraint fk_attendance_employee
        foreign key(employee_id)
        references employees(employee_id)
);

create table devices (
    device_id text primary key,

    device_name text,

    location text,

    is_active boolean default true,

    created_at timestamptz default now()
);