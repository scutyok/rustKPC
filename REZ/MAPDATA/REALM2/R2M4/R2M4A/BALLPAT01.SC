Name "ballpat01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			512.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait

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
TurnTo				ballpos01
MoveTo				ballpos01
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
TurnTo				ballpos02
MoveTo				ballpos02
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
TurnTo				ballpos03
MoveTo				ballpos03
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
TurnTo				ballpos04
MoveTo				ballpos04
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
TurnTo				ballpos05
MoveTo				ballpos05
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
TurnTo				ballpos06
MoveTo				ballpos06
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
TurnTo				ballpos07
MoveTo				ballpos07
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
TurnTo				ballpos08
MoveTo				ballpos08
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

