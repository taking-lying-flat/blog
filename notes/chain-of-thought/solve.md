```python solve · 搜索主循环（节选） | GitHub
for step in range(task.steps):
    if args.method_generate == 'sample':
        new_ys = [
            get_samples(task, x, y, args.n_generate_sample,
                        prompt_sample=args.prompt_sample, stop=task.stops[step])
            for y in ys
        ]
    elif args.method_generate == 'propose':
        new_ys = [get_proposals(task, x, y) for y in ys]
    new_ys = list(itertools.chain(*new_ys))
    ids = list(range(len(new_ys)))
    if args.method_evaluate == 'vote':
        values = get_votes(task, x, new_ys, args.n_evaluate_sample)
    elif args.method_evaluate == 'value':
        values = get_values(task, x, new_ys, args.n_evaluate_sample)
    if args.method_select == 'sample':
        ps = np.array(values) / sum(values)
        select_ids = np.random.choice(
            ids, size=args.n_select_sample, p=ps
        ).tolist()
    elif args.method_select == 'greedy':
        select_ids = sorted(
            ids, key=lambda x: values[x], reverse=True
        )[:args.n_select_sample]
    select_new_ys = [new_ys[select_id] for select_id in select_ids]
    ys = select_new_ys
```
