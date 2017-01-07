import math
import git

class Config:
	execfile('./config.py')
	def get_ftime(self, time):
		return [self.sec_per_commit*(time-0.5), (self.sec_per_commit)*1.0*time - self.sec_per_stable*0.5, \
			(self.sec_per_commit)*1.0*time + self.sec_per_stable*0.5, self.sec_per_commit*(time+0.5)]
		
cfg = Config()

""" unused, funny but overcomplicated
def with_float_attr(func):
	def func_wrapper(*args, **kwargs):
		kwargs['attr_to_str'] = lambda f: '%.3f' % f
		return func(*args, **kwargs)
	return func_wrapper

def with_point_list(func):
	def func_wrapper(*args, **kwargs):
		kwargs['attr_to_str'] = lambda p: 'M ' + ' L '.map(lambda t: '%.2f %.2f' % t, p)
		return func(*args, **kwargs)
	return func_wrapper
"""

class AObject:
	""" Supports common attributes: opacity, fill, stroke
	stroke-dasharray as dlength, stroke-dashoffset as doffset """
	def __init__(self, tag_name):
		self.tag_name = tag_name
		self.class_name = self.__class__.__name__
		self.frames = {}
		self.content = ''
		self.static_attr = {}
		
	def add_static_attr(self, **kwargs):
		self.static_attr.update(kwargs)

	def add_frame(self, ftime, **kwargs):
		ktime = max(0, round(ftime*100))
		if not ktime in self.frames:
			self.frames[ktime] = kwargs
		else:
			self.frames[ktime].update(kwargs)

	def add_anim(self, ftimes, **kwargs):
		""" add_anim([0 ,0.5, 2.3], fill=['green','blue','green'], r=[3, 5, 3]) """
		# more pythonic way?
		for i,ft in enumerate(ftimes):
			fv = [v[i] for v in kwargs.values()]
			d = dict(zip(kwargs.keys(), fv))
			self.add_frame(ft, **d)

	def get_frame_values(self, attr_name):
		""" Sorted (time,value) list for name """
		return [(f/100.0, self.frames[f][attr_name]) for f in sorted(self.frames.keys()) if attr_name in self.frames[f]]

	def animate_attr(self, name, attr_to_str = (lambda f: '%.2f' % f), anim_tag = 'animate'):
		names = name.split(':')
		prop_name = names[0]
		attr_name = names[-1]

		res = ''
		fvs = self.get_frame_values(prop_name)
		if not fvs:
			return res

		prev = (max(0,fvs[0][0] - 0.01), None)
		calc = 'calcMode="discrete"'
		for  fv in fvs:
			if fv[1] != prev[1]:
				res += '  <%s attributeName="%s" begin="%.2fs" dur="%.2fs" to="%s" fill="freeze" %s/>\n' \
					% (anim_tag, attr_name, prev[0], max(0.01, fv[0] - prev[0]), attr_to_str(fv[1]), calc)
			prev = fv
			calc = ''
		return res

	def animate(self):
		res = ''.join(self.animate_attr(a) for a in ['opacity', 'dlength:stroke-dasharray', 'doffset:stroke-dashoffset']) + \
			''.join(self.animate_attr(a, attr_to_str = str) for a in ['fill', 'stroke'])
		return res

	def render(self):
		svg = '<%s class="%s" ' % (self.tag_name, self.class_name)
		svg += ''.join('%s="%s" ' % (k,v) for k,v in self.static_attr.items()) + '>\n'
		svg += self.animate()
		svg += self.content
		svg += '</%s>\n' % self.tag_name
		return svg

class AText(AObject):
	def __init__(self, content):
		AObject.__init__(self, 'text')
		self.content = content

class ANone(AObject):
	""" Dummy object to spare some if statements elsewhere """
	def __init__(self):
		AObject.__init__(self, '')
	def add_frame(self, ftime, **kwargs):
		pass
	def add_anim(self, ftimes, **kwargs):
		pass
	def get_frame_values(self, attr_name):
		return []
	def animate_attr(self, name, **kwargs):
		return ''
	def animate(self):
		return ''
	def render(self):
		return ''

class ADot(AObject):
	""" Animated dot, supports x, y, r """

	def __init__(self):
		AObject.__init__(self, 'circle')

	def animate(self):
		return AObject.animate(self) + ''.join(map(self.animate_attr, ['x:cx', 'y:cy', 'r']))

	@staticmethod
	def get_length(r):
		return 2*r*math.pi

	def test(self):
		self.add_frame(2.3, x=125)
		self.add_frame(1.3, x=115, y=25, r=10)
		self.add_frame(4.3, x=225, y=35)
		self.add_frame(4.0, fill='blue')
		self.add_frame(4.5, fill='yellow')
		self.add_frame(5.0, fill='purple')
		self.add_anim([5.0, 6.0, 7.0, 8.0], r=[10, 20, 20, 10], opacity=[1.0, 0.0, 0.0, 1.0])

		len = ADot.get_length(10)
		self.add_frame(8, stroke = 'white') 
		self.add_anim([8, 10], dlength = [len/2, len/2], doffset = [0, 10* len])
		return self.render()

class APoly(AObject):
	""" Animated poly-line, supports 'points': list of 2-tuples (x and y) """

	def __init__(self):
		AObject.__init__(self, 'path')

	def animate(self):
		def point_list_to_str(p):
			return 'M ' + ' L '.join('%.2f %.2f' % t for t in p)
		return AObject.animate(self) + self.animate_attr('points:d', attr_to_str = point_list_to_str)

	@staticmethod
	def get_length(points):
		l = reduce(lambda r,p: (p[0], p[1], r[2] + ((r[0]-p[0])**2 + (r[1]-p[1])**2)**0.5),
			points[1:], (points[0][0], points[0][1], 0))
		return l[2]

	def test(self):
		self.add_frame(0.0, points=[ (100,100), (200,100), (300, 100) ] )
		self.add_frame(1.0, points=[ (100,150), (200,100), (300, 150) ] )
		self.add_frame(1.7, points=[ (100,150), (300,100), (400, 150) ] )
		len = APoly.get_length([ (100,150), (300,100), (400, 150) ])
		self.add_frame(2, dlength = len, doffset = 0)
		self.add_anim([3, 4], doffset = [len, 0])
		return self.render()

class AArc(AObject):
	""" Animated arc with fixed radius, supports 'points': list of 2-tuples (x,y) of length 2  """

	def __init__(self, r):
		AObject.__init__(self, 'path')
		self.r = r

	def animate(self):
		def arc_points_to_str(p):
			return 'M ' + (' A %.2f %.2f ' % (self.r,self.r) + 3*'0 ').join('%.2f %.2f' % t for t in p[:2])
		return AObject.animate(self) + self.animate_attr('points:d', attr_to_str = arc_points_to_str)

	def test(self):
		self.r = 100
		self.add_anim([0.0, 0.5, 1.0, 2.0], points = [\
				[ (100,100), (300, 100) ], \
				[ (100,100), (300, 150) ], \
				[ (200,100), (300, 150) ], \
				[ (200,50),  (300, 200) ]],
			stroke = [ 'black', 'cyan', 'cyan', 'orange'])
		return self.render()

class AGroup(AObject):
	""" Transformable group, supports 'scale' (scalar ratio), 'rotate' (scalar angle in degrees), 'translate' (2-tuple: x,y)
	Members added to this group are rendered by the group. Animations for members are still applied. """

	def __init__(self):
		AObject.__init__(self, '')
		self.members = []

	def add_member(self, m):
		self.members.append(m)

	def animate(self):
		return ''

	def render(self):
		fmt = 'animateTransform additive="sum" type="{0}"'
		svg = '<g>\n'
		svg += self.animate_attr('translate:transform', anim_tag = fmt.format('translate'), attr_to_str = lambda v: '%.2f %.2f' % v)
		svg += '<g>\n'+'<g>\n'.join(self.animate_attr('%s:transform' % a, anim_tag = fmt.format(a)) for a in ['rotate', 'scale'])
		svg += ''.join(m.render() for m in self.members)
		svg += '</g>\n' * 3
		return svg

	def test(self):
		d = ADot()
		d.add_frame(0, x = 10, y = 10, r = 10)
		self.add_member(d)
		d = ADot()
		d.add_frame(0, x = 60, y = 60, r = 10)
		self.add_member(d)
		t = AText('hello')
		t.add_static_attr(**{'text-anchor':'end'})
		self.add_member(t)

		self.add_frame(4.0, scale=1.0)
		self.add_frame(5.0, scale=0.1)
		self.add_frame(8.0, scale=0.8)
		self.add_frame(0.0, translate=(0,50))
		self.add_frame(6.0, translate=(200,50))
		self.add_frame(8.0, translate=(200,400))
		return self.render()


class DrawPos:
	""" Computed time-dependent position for drawing a node or something at a given time """
	def __init__(self):
		self.x = 0
		self.y = 0

class NodeDrawPos(DrawPos):
	""" With some extra helper variables """
	def __init__(self):
		DrawPos.__init__(self)
		self.width = 1
		self.draw_a1 = cfg.min_angle
		self.draw_a2 = cfg.max_angle
		self.draw_r = 0

class ChangedFile:

	def __init__(self, parent, path):
		self.parent = parent
		self.path = path
		self.node = None
		self.idx = 0

	def draw(self):
		def get_connector(lpos, npos):
			# 4-point connector poly-line between 2 endpoints
			return [(lpos[0] + cfg.conn_dx, lpos[1]), (lpos[0] + cfg.conn_turn_dx, lpos[1]),
				(lpos[0] + cfg.conn_turn_dx + math.fabs(npos[1] - lpos[1])*cfg.conn_slope, npos[1]),
				(npos[0] - cfg.conn_r, npos[1])]

		# prepare filename and transform group
		time = self.parent.time
		nfiles = len(self.parent.modified_files)
		file_label = AText(self.path if len(self.path) < 30 else '...'+self.path[-27:])
		file_label.add_static_attr(**{'text-anchor':'end'})

		file_group = AGroup()
		file_group.add_member(file_label)

		# prepare connector poly-line
		connector = APoly()
		connector.class_name = 'connector'

		# need stable-state times (prev -> cur -> next)...
		cur_ft = cfg.get_ftime(time)
		# cur_ft[1:3] = [cur_ft[1] + 0.1*self.idx, cur_ft[2] + 0.1*self.idx]
		prev_ft = cfg.get_ftime(time-1)
		next_ft = cfg.get_ftime(time+1)
		stab_ft = [prev_ft[2]] + cur_ft[1:3] + [next_ft[1]]

		# ... and positions for filename label
		cur_label_pos = (cfg.file_x, cfg.canvas_height/2 - (nfiles/2 - self.idx + 1) * cfg.file_dy)
		prev_label_pos = (self.node.pos[time-1].x, self.node.pos[time-1].y)
		next_label_pos = (self.node.pos[time+1].x, self.node.pos[time+1].y)

		# shift and scale label
		file_group.add_anim(stab_ft, \
			translate = [prev_label_pos] + 2*[cur_label_pos] + [next_label_pos], \
			scale = [0.1, 1, 1, 0.1])
		file_label.add_anim(stab_ft, opacity = [0, 1, 1, 0])

		# connector points computed from stable label points and stable node points
		dp = self.node.pos[time]
		cp = get_connector(cur_label_pos, (dp.x, dp.y))
		
		# animated points
		# connector.add_anim(stab_ft, points = [4*[prev_label_pos], cp, cp, 4*[next_label_pos]] )
		# connector.add_anim([0] + cur_ft, opacity = [0, 0, 1, 1, 0])

		# stable points + drawing anim
		conn_ft = [cur_ft[1] + cfg.conn_sec - 0.01, cur_ft[1] + cfg.conn_sec,
			cur_ft[2] - cfg.conn_sec, cur_ft[2] - cfg.conn_sec + 0.01]
		clen = APoly.get_length(cp);
		connector.add_frame(cur_ft[1], points = cp, dlength = clen)
		connector.add_anim([0, cur_ft[1], cur_ft[1]+0.01, cur_ft[2], cur_ft[2]+0.01], opacity = [0, 0, 1, 1, 0])
		connector.add_anim([cur_ft[1], conn_ft[1], conn_ft[2], cur_ft[2]], doffset = [clen, 0, 0, clen])

		# animated selection circle
		conn_circle = ADot()
		conn_circle.class_name = 'connector-circle'
		clen = ADot.get_length(cfg.conn_r)
		conn_circle.add_frame(conn_ft[0], x = cp[3][0] + cfg.conn_r, y = cp[3][1], r = cfg.conn_r, dlength = clen/8)
		conn_circle.add_anim([0] + conn_ft, opacity = [0, 0, 1, 1, 0])
		conn_circle.add_anim(conn_ft[1:3], doffset = [0, 2*cfg.sec_per_stable*clen])

		return [file_group, connector, conn_circle]

class Commit:

	def __init__(self, gitcommit, time):
		self.commit = gitcommit
		self.time = time

		# store modified files by this commit (compared to first parent commit)
		di = gitcommit.diff(gitcommit.parents[0])
		mfiles = \
			[ChangedFile(self, dd.a_path) for dd in di.iter_change_type('M')] + \
			[ChangedFile(self, dd.b_path) for dd in di.iter_change_type('A')] + \
			[ChangedFile(self, dd.b_path) for dd in di.iter_change_type('R')] + \
			[ChangedFile(self, dd.a_path) for dd in di.iter_change_type('D')]
		self.modified_files = sorted(mfiles, key = lambda f: f.path)
		for i,f in enumerate(self.modified_files):
			f.idx = i+1

	def add_to_tree(self, tree):
		" Add modified files to tree, at a given time "

		for f in self.modified_files:
			f.node = tree.add_node(f.path.split('/'),  self.time)

		# add some unimportant folder to get an idea about the whole source tree
		def add_tree_to_tree(t, max_depth = 4):
			if not max_depth: return
			for subt in t.trees:
				if not subt.trees: continue
				tree.add_node(subt.path.split('/'), -1)
				add_tree_to_tree(subt, max_depth - 1)
		add_tree_to_tree(self.commit.tree, -1)

	def get_label_text(self):
		author_names = self.commit.author.name.split(' ')
		res = '%s by %s' % (self.commit.hexsha[:8], author_names[0])
		try: res += ' ' + author_names[1][0]
		except: pass
		return res
	
	def get_detail_text(self):
		str = '%s by %s at %s' % (self.commit.hexsha[:8], self.commit.author.name, \
			self.commit.authored_datetime.strftime('%x %X'))
		res = (str if len(str) < 60 else str[:57] + '...')
		str = self.commit.message.split('\n')[0]
		res += '<tspan x="%d" dy="%d">' % (cfg.detail_x, cfg.detail_dy)
		res += (str if len(str) < 60 else str[:57]+'...') + '</tspan>'
		return res

	def svg_draw(self):
		commit_label = AText(self.get_label_text())
		commit_label.add_static_attr(x = cfg.commit_x, y = cfg.commit_y + (self.time-1)*cfg.commit_dy)
		
		commit_details = AText(self.get_detail_text())
		commit_details.add_static_attr(x = cfg.detail_x, y = cfg.detail_y, style='opacity:0')
		commit_details.add_anim(cfg.get_ftime(self.time), opacity = [0, 1, 1, 0])

		file_obj_list = reduce(lambda a,b: a+b, (f.draw() for f in self.modified_files), [])
		return [commit_label, commit_details] + file_obj_list

class TreeNode:
	" Node in the source tree with time-invariant and time-dependent properties "
			
	def __init__(self,  parent,  name):

		# some main properties
		self.parent = parent
		self.name = name
		self.children = {}

		# numeric subtree properties not chaning in time
		self.level = (1 if parent  is None else parent.level+1)
		self.depth = 1
		self.n_leaves = 1
		self.n_nodes = 1

		# first and last child in order
		self.min_child = None
		self.max_child = None

		# important times (when the subtree is changed)
		self.times = set()
		# states computed for any given times
		self.pos = {}
		self.conn_pos = {}

	def add_node(self,  name_list,  time):
		" Add sequence of nodes with a path as name-list, recording time as important "

		# create/locate child with first name in list
		child_name = name_list[0]
		if not  child_name in self.children:
			self.children[child_name] = TreeNode(self,  child_name)
		ch = self.children[child_name]

		# record important time and continue adding remaining names
		if time != -1: ch.times.add(time)
		if len(name_list) > 1:
			return ch.add_node(name_list[1:],  time)
		else:
			return ch

	def calc_size(self):
		" Compute time-invariant numeric properties "

		if self.children:
			# visit children, get max depth, sum n_nodes and n_leaves
			self.depth,self.n_nodes,self.n_leaves = reduce( \
				lambda r,ch: (max(r[0], ch[0]), r[1]+ch[1], r[2]+ch[2]), \
				(ch.calc_size() for ch in self.children.values()), \
				(0,1,0))
			self.depth += 1
			# firs/last child
			ch_names = sorted(self.children.keys())
			self.min_child = self.children[ch_names[0]]
			self.max_child = self.children[ch_names[-1]]
		else:
			# leaf
			self.depth,self.n_nodes,self.n_leaves = 1,1,1
			self.min_child,self.max_child = self,self

		return (self.depth, self.n_nodes, self.n_leaves)

	def calc_width(self, time):
		" Compute width of node/subtree, for a state at a given time "

		# create the state if needed
		if time not in self.pos.keys():
			self.pos[time] = NodeDrawPos()
		pos = self.pos[time]

		if self.children:
			# visit children
			pos.width = sum(ch.calc_width(time) for ch in self.children.values())
		else:
			# leaf: decide importance
			pos.width = (10 if time in self.times and cfg.dyn_angle else 1)
		return pos.width

	def calc_pos(self,  time, max_depth = 0, draw_a = 0):
		" Compute drawing properties from all widths, for a state at a given time "

		pos = self.pos[time]
		ppos = None
		if self.parent:
			ppos = self.parent.pos[time]
			# compute own angle range
			pos.draw_a1 = draw_a
			pos.draw_a2 = pos.draw_a1 + (ppos.draw_a2 - ppos.draw_a1) * pos.width/ppos.width
		else:
			# full angle range
			pos.draw_a1 = 1.0 * cfg.min_angle
			pos.draw_a2 = 1.0 * cfg.max_angle
			max_depth = self.depth - 1

		# radius

		""" option 1
		if not self.times:
			pos.draw_r = cfg.radius * (self.level - 1) / max_depth
		elif self.depth != 1:
			pos.draw_r = cfg.radius * (max_depth - self.depth + 1) / max_depth
		else:
			pos.draw_r = cfg.radius * (1.1 if time in self.times and cfg.dyn_rad else 1)
		"""

		""" option 2
		if not time in self.times or self.depth != 1:
			pos.draw_r = cfg.radius * (self.level - 1) / max_depth
		else:
			pos.draw_r = cfg.radius * (1.0 if time in self.times and cfg.dyn_rad else 1)
		"""

		""" option 3 """
		if self.parent and not self.times:
			pos.draw_r = ppos.draw_r + cfg.radius * 1.0/max_depth
		elif self.depth != 1:
			pos.draw_r = cfg.radius * (max_depth - self.depth + 1) / max_depth
		else:
			pos.draw_r = cfg.radius * (1.1 if time in self.times and cfg.dyn_rad else 1)

		cx = cfg.canvas_width
		cy = 0.5*cfg.canvas_height

		# store cartesian coords too (at middle of angle range)
		phi = math.radians(0.5*(pos.draw_a1 + pos.draw_a2))
		pos.x = cx + pos.draw_r * math.cos(phi)
		pos.y = cy - pos.draw_r * math.sin(phi)

		# connector to parent
		self.conn_pos[time] = DrawPos()
		if ppos:
			self.conn_pos[time].x = cx + ppos.draw_r * math.cos(phi)
			self.conn_pos[time].y = cy - ppos.draw_r * math.sin(phi)

		# visit children and let them compute props
		draw_a = pos.draw_a1
		for i in sorted(self.children.keys()): 
			child = self.children[i]
			child.calc_pos(time, max_depth, draw_a)
			draw_a = child.pos[time].draw_a2

	def svg_draw_circular(self):
		
		sec = 1.0 * cfg.sec_per_commit

		# svg object to animate
		dot = ADot()
		conn = APoly() if self.parent else ANone()
		arc = AArc(self.pos[0].draw_r) if self.parent and self.min_child != self.max_child else ANone()

		# init colors
		for obj,attr in [(dot, 'fill'), (conn, 'stroke'), (arc, 'stroke')]:
			obj.add_frame(0, **{attr: cfg.tree_base_color})

		for time in self.pos.keys():
			ft = cfg.get_ftime(time)

			# positions
			dot.add_anim(ft[1:3], x = 2*[self.pos[time].x], y = 2*[self.pos[time].y], \
				r = 2*[cfg.dot_high_r if time in self.times and not self.children else cfg.dot_base_r])
			conn.add_anim(ft[1:3], points = 2*[ \
				[(p[time].x,p[time].y) for p in (self.pos, self.conn_pos)]])
			arc.add_anim(ft[1:3], points = 2*[ \
				[(ch.conn_pos[time].x,ch.conn_pos[time].y) for ch in (self.min_child, self.max_child)]])

			# colors
			color = (cfg.tree_high_color if time in self.times else cfg.tree_base_color)
			for obj,attr in [(dot, 'fill'), (conn, 'stroke'), (arc, 'stroke')]:
				obj.add_anim(ft[1:3], **{attr: 2*[color]})

		# children
		return reduce(lambda a,b: a+b, (ch.svg_draw_circular() for ch in self.children.values()),[]) + [dot,conn,arc]

class GitLogDrawer:
	""" Build and draw source tree from commit history """

	def __init__(self,  repo_path,  branch_name):
		self.repo_path = repo_path
		self.branch_name = branch_name
		self.tree = TreeNode(None, '*' )
		self.commits = {}

	def build(self):
		# init repository object
		repo = git.Repo(self.repo_path)

		# iterate backwards in the commit history of the given branch (time: 1..count)
		time = 1 # cfg.max_commits
		for c in repo.iter_commits(self.branch_name, max_count = cfg.max_commits):
			# build tree and store changed files
			self.commits[time] = Commit(c, time)
			self.commits[time].add_to_tree(self.tree)

			# print top-level tree objects at this commit
			# print(reduce(lambda a, b: a + ' ' + b,  (t.path for t in c.tree),  'root tree [') + ']')

			time = time+1

		# time: 0 ... count+1, 0/count+1 meaning state without commit ( -> and without important files)
		self.tree.calc_size()
		for time in range(0, cfg.max_commits+2):
			self.tree.calc_width(time)
			self.tree.calc_pos(time)

	def draw(self):

		# svg-display of tree built from modified files
		print('<html> <head> <link rel="stylesheet" type="text/css" href="svgstyle.css"> </head>')
		print('<body> <svg width="%s" height="%s">' % (cfg.canvas_width, cfg.canvas_height))
		for line in open('./back.svg'):
			print(line)

		# commit frame
		commit_frame = AObject('rect')
		commit_frame.class_name = 'commit_frame'
		commit_frame.add_static_attr(x = cfg.commit_frame_dx, y = -cfg.commit_frame_h*0.5,
			width = cfg.commit_frame_w, height = cfg.commit_frame_h, style='opacity:0')
		commit_frame.add_anim(cfg.get_ftime(1)[0:2], opacity = [0, 1])
		commit_frame.add_anim(cfg.get_ftime(len(self.commits))[2:4], opacity = [1, 0])

		commit_g = AGroup()
		commit_g.add_member(commit_frame)
		commit_g.add_frame(0, translate = (cfg.commit_x, cfg.commit_y))
		for c in self.commits.values():
			commit_g.add_anim( cfg.get_ftime(c.time)[1:3], translate = 
				2*[(cfg.commit_x, cfg.commit_y + cfg.commit_dy*(c.time-1))])

		scene = [commit_g]
		scene += self.tree.svg_draw_circular()
		scene += reduce(lambda a,b: a+b, (obj.svg_draw() for obj in self.commits.values()), [])
		for obj in scene:
			print(obj.render())

		print('</svg> </body> </html>')

if __name__ == '__main__':

	drawer = GitLogDrawer(cfg.repo_path, cfg.branch_name)
	drawer.build()
	drawer.draw()

	"""
	print('<html> <head> <link rel="stylesheet" type="text/css" href="svgstyle.css"> </head>')
	print('<body> <svg width="%s" height="%s">' % (cfg.canvas_width, cfg.canvas_height))

	print(ADot().test())
	print(APoly().test())
	print(AArc(50).test())
	print(AGroup().test())

	print('</svg> </body> </html>')
	"""
