Name "manlib02a"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			1024.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait
Goto Loop3

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK1		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK1		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos01
MoveTo				ballwalk01pos01
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop2

///////////////////////////////////////////////////////////////////////////////

Loop2:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK2		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK2		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos02
MoveTo				ballwalk01pos02
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop3

///////////////////////////////////////////////////////////////////////////////

Loop3:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK3		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK3		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos03
MoveTo				ballwalk01pos03
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop4

///////////////////////////////////////////////////////////////////////////////

Loop4:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK4		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK4		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos04
MoveTo				ballwalk01pos04
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop5

///////////////////////////////////////////////////////////////////////////////

Loop5:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK5		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK5		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos05
MoveTo				ballwalk01pos05
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop6

///////////////////////////////////////////////////////////////////////////////

Loop6:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK6		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK6		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos06
MoveTo				ballwalk01pos06
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop7

///////////////////////////////////////////////////////////////////////////////

Loop7:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK7		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK7		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos07
MoveTo				ballwalk01pos07
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop8

///////////////////////////////////////////////////////////////////////////////

Loop8:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK8		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK8		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				ballwalk01pos08
MoveTo				ballwalk01pos08
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0
Goto 	Loop1


///////////////////////////////////////////////////////////////////////////////
//	do the meanie, attacking stuff here
///////////////////////////////////////////////////////////////////////////////

Attack1:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop1
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget   	
TurnTo		ballwalk01pos01
Run
StartAnim	"Run"
MoveTo		ballwalk01pos01

Goto	Attack1

Attack2:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop2
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos02
Run
StartAnim	"Run"
MoveTo		ballwalk01pos02
Goto	Attack2

Attack3:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop3
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos03
Run
StartAnim	"Run"
MoveTo		ballwalk01pos03
Goto	Attack3

Attack4:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop4
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos04
Run
StartAnim	"Run"
MoveTo		ballwalk01pos04
Goto	Attack4

Attack5:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop5
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos05
Run
StartAnim	"Run"
MoveTo		ballwalk01pos05
Goto	Attack5

Attack6:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop6
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos06
Run
StartAnim	"Run"
MoveTo		ballwalk01pos06
Goto	Attack6

Attack7:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop7
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos07
Run
StartAnim	"Run"
MoveTo		ballwalk01pos07
Goto	Attack7

Attack8:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop8
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget 
TurnTo		ballwalk01pos08
Run
StartAnim	"Run"
MoveTo		ballwalk01pos08
Goto	Attack8


///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Crazy:
	LoopAnim	FALSE
	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		Stand
	TurnToTarget			0.57	
	RangedAttackTarget
	MeleeAttackTarget 
	Goto				Crazy

///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Stand:
	LoopAnim	FALSE
	OnDamage_Goto			crazy
	OnEnemySighted_Goto		crazy
	TurnToTarget			0.57	
	Goto				stand

